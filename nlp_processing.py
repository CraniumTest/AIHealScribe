from transformers import pipeline

def process_text(transcription):
    nlp = pipeline('fill-mask', model='bert-base-uncased')
    return nlp(transcription)
