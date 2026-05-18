import json
import re
import logging
import time
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

logger = logging.getLogger(__name__)

EXTRA_STOPWORDS = frozenset({'si', 'mi', 'ci', 'vi', 'ti', 'lo', 'la', 'li', 'le', 'ne', 'cosa', 'po'})


async def sentiment_analyze(sentiment_analyzer, hate_analyzer, emotion_analyzer):
    with open('messages.json', 'r', encoding='utf-8') as f:
        messages = json.load(f)

    analyzed = []
    logger.info("analyzing today's messages...")

    for msg in messages:
        text = msg.get('text', '')
        if len(text.strip()) < 3:
            continue
        sentiment = sentiment_analyzer.predict(text)
        hatefulness = hate_analyzer.predict(text)
        emotion = emotion_analyzer.predict(text)
        analyzed.append({
            'id':               msg['id'],
            'text':             text,
            'date':             msg['date'],
            'sentiment_probas': sentiment.probas,
            'hate_probas':      hatefulness.probas,
            'emotion_probas':   emotion.probas,
        })

    with open('sentiment.json', 'w', encoding='utf-8') as f:
        json.dump(analyzed, f, indent=2, ensure_ascii=False)

    logger.info("done: %d messages analyzed", len(analyzed))
    return analyzed


def _load_spacy():
    import spacy
    # will raise OSError if model not installed, run: python -m spacy download it_core_news_sm
    return spacy.load('it_core_news_sm', disable=['parser', 'ner'])


def _preprocess(telegram_posts: list[str], nlp) -> list[str]:
    # only nouns and adjectives, verbs pollute topics with italian light verbs (fare, dare, vedere...)
    cleaned = []
    for doc in nlp.pipe(telegram_posts, batch_size=50):
        tokens = [
            token.lemma_.lower() for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.like_url
            and not token.like_email
            and token.pos_ in ('NOUN', 'ADJ')
            and len(token.lemma_) > 2
            and token.lemma_.lower() not in EXTRA_STOPWORDS
        ]
        cleaned.append(' '.join(tokens))
    return cleaned


def _topic_nmf(telegram_posts: list[str], n_topics: int = 5, nlp=None) -> dict:
    start = time.time()

    cleaned_posts = _preprocess(telegram_posts, nlp)
    topic_candidates = [p for p in cleaned_posts if len(p.split()) >= 2]

    if len(topic_candidates) < 5:
        logger.warning("not enough posts for NMF (%d)", len(topic_candidates))
        return {'method': 'nmf', 'topics': {}, 'error': 'insufficient_data'}

    vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.85)

    try:
        tfidf_matrix = vectorizer.fit_transform(topic_candidates)
    except ValueError as e:
        logger.error("vectorizer failed: %s", e)
        return {'method': 'nmf', 'topics': {}, 'error': 'vectorizer_failed'}

    n_topics = min(n_topics, tfidf_matrix.shape[0] - 1)
    model = NMF(n_components=n_topics, init='nndsvda', random_state=42, max_iter=200)
    model.fit(tfidf_matrix)

    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for i, component in enumerate(model.components_):
        top_indices = component.argsort()[-5:][::-1]
        raw_keywords = [feature_names[j] for j in top_indices]

        # cheap stem dedup instead of nltk snowball (saves ram on rpi)
        seen_stems, keywords = set(), []
        for word in raw_keywords:
            if word[:5] not in seen_stems:
                keywords.append(word)
                seen_stems.add(word[:5])

        topics[f'topic_{i}'] = ', '.join(keywords)

    elapsed = round(time.time() - start, 1)
    logger.info("nmf done in %ss", elapsed)

    return {
        'method': 'nmf',
        'elapsed_seconds': elapsed,
        'topics': topics,
        'n_docs': len(topic_candidates),
    }


async def analyze_daily_topics(sentiment_data: list[dict], n_topics: int = 5) -> dict:
    telegram_posts = [
        item['text'] for item in sentiment_data
        if item.get('text') and len(item['text']) > 7
    ]

    logger.info("starting topic analysis: %d posts", len(telegram_posts))

    nlp = _load_spacy()
    analysis_result = _topic_nmf(telegram_posts, n_topics=n_topics, nlp=nlp)

    emotion_clusters: dict[str, list[str]] = defaultdict(list)
    for item in sentiment_data:
        if item.get('emotion_probas') and item.get('text') and len(item['text']) > 7:
            dominant = max(item['emotion_probas'], key=item['emotion_probas'].get)
            emotion_clusters[dominant].append(item['text'])

    emotion_topics = {}
    for emotion, posts in emotion_clusters.items():
        if len(posts) < 5:
            emotion_topics[emotion] = 'N/A'
            continue
        cleaned = _preprocess(posts, nlp)
        try:
            tfidf = TfidfVectorizer(min_df=1)
            tfidf_matrix = tfidf.fit_transform(cleaned)
            word_weights = tfidf_matrix.sum(axis=0)
            ranked = sorted(
                tfidf.vocabulary_.items(),
                key=lambda x: word_weights[0, x[1]],
                reverse=True
            )
            emotion_topics[emotion] = ', '.join([w for w, _ in ranked[:5]])
        except ValueError as e:
            logger.error("keyword extraction failed for %s: %s", emotion, e)
            emotion_topics[emotion] = 'N/A'

    analysis_result['emotion_topics'] = emotion_topics

    with open('topics.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)

    logger.info("topic analysis complete: %s", analysis_result)
    return analysis_result