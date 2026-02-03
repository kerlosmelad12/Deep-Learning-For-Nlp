import nltk
text="115712 I understand I would like to assist you We would need to get you into a private secured link to further assist"

for w, pos in nltk.pos_tag(text.split()):
    print(w,pos[0])
