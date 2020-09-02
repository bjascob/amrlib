from   unidecode import unidecode


# Solution to manually replace CP1252 characters with replacements and then convert to ASCII
# This is setup to load and filter out bad characters from the raw AMR LDC2020T02 corpus
# This will return all entries in ASCII
def load_raw_amr(fname):
    with open(fname, 'rb') as f:
        raw_bytes = f.read()
    data = to_ascii(raw_bytes)
    # Strip off non-amr header info (see start of Little Prince corpus)
    lines = [l for l in data.splitlines() if not (l.startswith('#') and not l.startswith('# ::'))]
    data = '\n'.join(lines)
    # Split into different entries based on a space (ie.. extra linefeed) between the graph and text
    entries = data.split('\n\n')            # split via standard amr
    entries = [e.strip() for e in entries]  # clean-up line-feeds, spaces, etc
    entries = [e for e in entries if e]     # remove any empty entries
    return entries


# From https://stackoverflow.com/questions/6609895/efficiently-replace-bad-characters
convert_dict = {
    b'\xc2\x82' : b',',        # High code comma
    b'\xc2\x84' : b',,',       # High code double comma
    b'\xc2\x85' : b'...',      # Tripple dot
    b'\xc2\x88' : b'^',        # High carat
    b'\xc2\x91' : b"'",        # Forward single quote - '\x27'
    b'\xc2\x92' : b"'",        # Reverse single quote - '\x27'
    b'\xc2\x93' : b'"',        # Forward double quote - '\x22'
    b'\xc2\x94' : b'"',        # Reverse double quote - '\x22'
    b'\xc2\x95' : b' ',
    b'\xc2\x96' : b'-',        # High hyphen
    b'\xc2\x97' : b'--',       # Double hyphen
    b'\xc2\x99' : b' ',
    b'\xc2\xa0' : b' ',
    b'\xc2\xa6' : b'|',        # Split vertical bar
    b'\xc2\xab' : b'<<',       # Double less than
    b'\xc2\xbb' : b'>>',       # Double greater than
    b'\xc2\xbc' : b'1/4',      # one quarter
    b'\xc2\xbd' : b'1/2',      # one half
    b'\xc2\xbe' : b'3/4',      # three quarters
    b'\xca\xbf' : b"'",        # c-single quote - '\x27'
    b'\xcc\xa8' : b'',         # modifier - under curve
    b'\xcc\xb1' : b''          # modifier - under line
}
def to_ascii(raw_bytes):
    for code, repl in convert_dict.items():
        raw_bytes = raw_bytes.replace(code, repl)
    text = raw_bytes.decode('utf-8')
    text = unidecode(text)      # converts unicode to ASCII
    return text
