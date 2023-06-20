from nltk.tokenize import sent_tokenize
from nltk.translate.bleu_score import sentence_bleu


print("Average BLEU score:", sentence_bleu(["The toilet is next to the kitchen"], "The toilet is next to Room A13. Do you want me to take you there?"))