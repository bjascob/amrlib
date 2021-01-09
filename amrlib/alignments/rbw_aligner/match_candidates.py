from word2number import w2n


# Convert months to numeric values
# AMR graphs always use the integer value of the month, not the word
month2num = {'january':'1', 'february':'2', 'march':'3', 'april':'4', 'may':'5', 'june':'6',
             'july':'7', 'august':'8', 'september':'9', 'october':'10', 'november':'11',
             'december':'12', 'jan':'1', 'feb':'2', 'mar':'3', 'apr':'4', 'may':'5', 'jun':'6',
             'jul':'7', 'aug':'8', 'sep':'9', 'oct':'10', 'nov':'11', 'dec':'12'}


# Misc word with possible alternate's used in the graph
# List from AlignSpans code
# word2words = {'/':['slash'], ';':['and'], ':':['mean'], '!':['expressive'], '..':['expressive'],
#                '...':['expressive'], '....':['expressive'], '?':['interrogative'], '%':['percentage-entity'],
#                'able':['possible'], 'according':['say'], 'also':['include'], 'anti':['oppose','counter'],
#                'anyway':['have-concession'], 'as':['same'], 'because':['cause'], 'but':['contrast','have-concession'],
#                'can':['possible'], 'cant':['possible'], "can't":['possible'], 'choice':['choose'],
#                'could':['possible'], 'death':['die'], 'French':['france','France'], 'french':['france','France'],
#                'have':['obligate'], 'her':['she'], 'his':['he'], 'how':['amr-unknown'], 'if':['cause'],
#                'illegal':['law'], 'like':['resemble'], 'life':['live'], 'may':['possible'], 'me':['i'],
#                'might':['possible'], 'my':['i'], "n't":['-'], 'no':['-'], 'non':['-'], 'not':['-'],
#                'of':['include','have-manner'], 'ok':['okay'], 'o.k.':['okay'], 'our':['we'], 'people':['person'],
#                'similar':['resemble'], 'since':['cause'], 'should':['recommend'], 'so':['infer','cause'],
#                'speech':['speak'], 'statement':['state'], 'them':['they'], 'these':['this'], 'those':['that'],
#                'thought':['think'], 'thoughts':['think'], 'uni':['university'], 'well':['good'], 'what':['amr-unknown'],
#                'who':['amr-unknown']}

# Original AlignWords list (subset of above) - this gives higher scores
word2words = {';':['and'], 'also':['include'], 'anti':['oppose','counter'], 'but':['contrast'],
              'because':['cause'], 'if':['cause'], 'no':['-'], 'not':['-'], 'of':['include'],
              'speech':['speak'], 'statement':['state']}
# Additional special cases
word2words['me'] = ['i']


# Return a list of possible word candidates in the graph to match for a word/lemma in the sentence
def get_match_candidates(token, lemma):
    # Obvious candidates
    word = token.lower()
    candidates = [word, lemma.lower()]
    # Get candidates for numbers
    try:
        number = w2n.word_to_num(word)
        candidates.append(str(number))
    except ValueError:
        pass
    # Get candidates for months
    month = month2num.get(word, None)
    if month is not None:
        candidates.append(month)
    # Get misc alternate word candidates
    candidates += word2words.get(word, [])
    # Check for words that are negative (graphs should include '-') and get the
    # non-negative version of the word
    if word.startswith('in') or word.startswith('un') or word.startswith('ir'):
        candidates.append( word[2:] )
    return candidates
