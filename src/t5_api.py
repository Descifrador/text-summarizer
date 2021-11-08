# import transformers
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def t5_summary(INCONTEXT: str) -> str:
    """
    Summary of T5 model.
    
    """
    # Load T5 model
    model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
    tokenizer = AutoTokenizer.from_pretrained('t5-base')

    ARTICLE = INCONTEXT
    inputs = tokenizer("summarize" + ARTICLE,
                       return_tensors="pt",
                       max_length=512,
                       truncation=True,
                       padding="max_length")
    outputs = model.generate(
        inputs["input_ids"],
        num_beams=4,
        length_penalty=2.0,
        max_length=150,
        min_length=50,
        no_repeat_ngram_size=3,
        early_stopping=True,
        num_return_sequences=1,
    )
    SUMMARY = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return SUMMARY
