# Styles CSS for chat blocks
def chat_block(speaker, content, is_question=True):
    # Define the color schemes for questions and answers
    background_color = "#303030" if is_question else "#1a1a1a"
    text_color = "#f8f8f8" if is_question else "#dcdcdc"
    shadow_color = "rgba(255, 255, 255, 0.05)" if is_question else "rgba(0, 0, 0, 0.05)"

    # Define header styles for speaker names
    head_style = f"font-size: 16px; font-weight: bold; color: {'#00BFFF' if is_question else '#FF4500'};"

    header = f"<span style='{head_style}'>{speaker}:</span>"

    # Styling for the entire chat block
    style = (
        f"background-color: {background_color}; "
        "border-radius: 10px; "
        "padding: 15px; "
        "margin: 10px 0; "
        "max-width: 100%; "
        "word-wrap: break-word; "
        f"box-shadow: 0px 4px 8px {shadow_color};"
        f"color: {text_color};"
    )

    return f"<div style='{style}'>{header} {content}</div>"

def header_content():
    header_style = "margin-bottom: 2em;"

    h1_style = """
        margin: 0;
        font-size: 32px;
        text-align: center;
        color: #007BFF;  
    """

    p_style = """
        margin: 10px 0 0;
        font-size: 18px;
        text-align: center;
        color: #666;  
    """

    h1 = f"<h1 style='{h1_style}'>My AI Assistant For AI Courses At École Centrale de Lyon</h1>"
    p = f"<p style='{p_style}'>Explore AI Course Materials and More</p>"

    return f"<header style='{header_style}'>{h1}{p}</header>"

def text2html(text, tag="p"):
    content = text.replace("&", "&amp;")
    content = content.replace("<", "&lt;")
    content = content.replace(">", "&gt;")
    content = content.replace('"', "&quot;")
    content = content.replace("'", "&#039;")
    content = content.replace("\n", "<br>")
    content = content.replace("\t", "&nbsp;" * 4)

    content = f"<{tag}>" + content + f"</{tag}>"

    return content
