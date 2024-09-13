import numpy as np
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy


# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, all_outputs):
        results = super().postprocess(
            all_outputs=all_outputs,
            aggregation_strategy=AggregationStrategy.FIRST,
        )
        return np.unique([result.get("word").strip() for result in results])


model_name = "./distilbert_keyphrase_extraction"
extractor = KeyphraseExtractionPipeline(model=model_name)

# # Inference
# text = """
# Keyphrase extraction is a technique in text analysis where you extract the
# important keyphrases from a document. Thanks to these keyphrases humans can
# understand the content of a text very quickly and easily without reading it
# completely. Keyphrase extraction was first done primarily by human annotators,
# who read the text in detail and then wrote down the most important keyphrases.
# The disadvantage is that if you work with a lot of documents, this process
# can take a lot of time.

# Here is where Artificial Intelligence comes in. Currently, classical machine
# learning methods, that use statistical and linguistic features, are widely used
# for the extraction process. Now with deep learning, it is possible to capture
# the semantic meaning of a text even better than these classical methods.
# Classical methods look at the frequency, occurrence and order of words
# in the text, whereas these neural approaches can capture long-term
# semantic dependencies and context of words in a text.
# """.replace(
#     "\n", " "
# )

# Book Tilte: Thinking, Fast and Slow
text = """
Major New York Times Bestseller. More than 2.6 million copies sold. One of The New York Times Book Reviews ten best books of the year. Selected by The Wall Street Journal as one of the best nonfiction books of the year. Presidential Medal of Freedom Recipient. Daniel Kahnemans work with Amos Tversky is the subject of Michael Lewiss bestselling The Undoing Project A Friendship That Changed Our Minds. . In his mega bestseller, Thinking, Fast and Slow, Daniel Kahneman, worldfamous psychologist and winner of the Nobel Prize in Economics, takes us on a groundbreaking tour of the mind and explains the two systems that drive the way we think.. . System 1 is fast, intuitive, and emotional System 2 is slower, more deliberative, and more logical. The impact of overconfidence on corporate strategies, the difficulties of predicting what will make us happy in the future, the profound effect of cognitive biases on everything from playing the stock market to planning our next vacationeach of these can be understood only by knowing how the two systems shape our judgments and decisions.. . Engaging the reader in a lively conversation about how we think, Kahneman reveals where we can and cannot trust our intuitions and how we can tap into the benefits of slow thinking. He offers practical and enlightening insights into how choices are made in both our business and our personal livesand how we can use different techniques to guard against the mental glitches that often get us into trouble. Topping bestseller lists for almost ten years, Thinking, Fast and Slow is a contemporary classic, an essential book that has changed the lives of millions of readers.
""".replace(
    "\n", " "
)
# truncate text to 256 tokens
text = " ".join(text.split(" ")[:256])

# # Book Tilte: Atomic Habits: An Easy & Proven Way to Build Good Habits & Break Bad Ones
# text = """
# Tiny Changes, Remarkable Results
# No matter your goals, Atomic Habits offers a proven framework for improving--every day.
# James Clear, one of the world’s leading experts on habit formation, reveals practical strategies
# that will teach you exactly how to form good habits, break bad ones, and master the
# tiny behaviors that lead to remarkable results. If you’re having trouble changing your habits,
# the problem isn’t you. The problem is your system. Bad habits repeat themselves again and again
# not because you don’t want to change, but because you have the wrong system for change.
# You do not rise to the level of your goals. You fall to the level of your systems. Here, you’ll
# get a proven system that can take you to new heights. Clear is known for his ability to distill
# complex topics into simple behaviors that can be easily applied to daily life and work. Here,
# he draws on the most proven ideas from biology, psychology, and neuroscience to create an
# easy-to-understand guide for making good habits inevitable and bad habits impossible.
# Along the way, readers will be inspired and entertained with true stories from Olympic gold
# medalists, award-winning artists, business leaders, life-saving physicians, and star comedians who
# have used the science of small habits to master their craft and vault to the top of their field.
# """.replace(
#     "\n", " "
# )


# # Book Tilte: The Courage to Be Disliked
# text = """
# Is happiness something you choose for yourself? The Courage to Be Disliked presents a simple and
# straightforward answer. Using the theories of Alfred Adler, one of the three giants of
# nineteenth-century psychology alongside Freud and Jung, this book follows an illuminating
# dialogue between a philosopher and a young man. Over the course of five conversations, the
# philosopher helps his student to understand how each of us is able to determine the direction of
# our own life, free from the shackles of past traumas and the expectations of others.
# Rich in wisdom, The Courage to Be Disliked will guide you through the concepts of self-forgiveness,
# self-care, and mind decluttering. It is a deeply liberating way of thinking, allowing you to
# develop the courage to change and ignore the limitations that you might be placing on yourself.
# This plainspoken and profoundly moving book unlocks the power within you to find lasting happiness
# and be the person you truly want to be. Millions have already benefited from its teachings, now you can too.
# """.replace(
#     "\n", " "
# )

keyphrases = extractor(text)

print(keyphrases)
