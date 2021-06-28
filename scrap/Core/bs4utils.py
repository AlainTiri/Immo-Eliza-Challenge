import bs4

def extract_text_from_tag(tag):
    """
    Extract all text data from a tag it's and children
    :param tag: beautifulSoup Tag
    :type tag: bs4.Tag
    :return: Single string concatenating all text
    :rtype: str
    """
    text = ''
    for element in tag.contents:
        if isinstance(element, bs4.Tag):
            text += extract_text_from_tag(element)
        elif isinstance(element, str):
            text += element
        else:
            raise TypeError(f"element was of type {type(element)}. Expecting str or bs4.Tag")
    text = text.replace("\t","")
    while text.find("\n\n") != -1:
        text = text.replace("\n\n", "\n")
    return text




