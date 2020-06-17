from wordcloud import WordCloud
import matplotlib.pyplot as plt


PATH_FONT = "/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf"


def create_wordcloud(text, path, width=900, height=500, *, font_path=PATH_FONT):
    """
    text: list of words
    """

    text = " ".join(text)
    wordcloud = WordCloud(background_color="white", font_path=font_path, width=900, height=500).generate(text)
    wordcloud.to_file(str(path))
