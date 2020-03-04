def normalize_string(string):
    """
    Lowercases and removes punctuation from input string, also strips the spaces from the borders and removes multiple spaces
    """
    string =re.sub(r"([,.!?'-:])", r"", string).lower()
    string = re.sub(' +', ' ',string).strip()
    return string

def do_this(display="",defaults_to_yes=True):
    """
    Parameters:
    -----------
    display: `str` to display to the user. Defaults to ""
    defaults_to_yes: `bool`: default answer if the user doesn't input anything

    Returns:
    --------
    True if the user wants to do it, else False.
    """
    answer=input("{} {}".format(display,"[Y]/N" if defaults_to_yes else "Y/[N]"))
    answer=answer.lower().strip()
    if defaults_to_yes:
        if answer != 'n' and answer != 'no':
            return True
        else:
            return False
    else:
        if answer != 'y' and answer != 'yes':
            return True
        else:
            return False
