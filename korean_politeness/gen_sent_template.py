adv = ["오늘은",	"내일은", "오후에",	"월요일에",	"화요일에",	"수요일에",	"목요일에",	"금요일에",	"토요일에",	"일요일에"]
n1 = ["아이들에게", "학생들에게", "동생들에게", "여동생에게", "남동생에게", "조카에게", "형에게", "누나에게", "사촌들에게", "아버지에게", "어머니에게", "할아버지에게", "할머니에게", "손님에게", "손님들에게", "고객님들에게"]
n2  = ["책을", "책들을", "잡지를", "소설을", "작품들을", "만화책을", "논문을", "논문들을"]

out = open('korean_minipair.txt', 'w')
for a in adv:
    for b in n1:
        for c in n2:
            out.write(a+' 제 '+b+' 올해 새로나온 '+c+' 신나게 읽어줘요 .'+'\n')
            out.write(a+' 제 '+b+' 올해 새로나온 '+c+' 신나게 읽어줘라 .'+'\n')
            out.write(a+' 제 '+b+' 올해 새로나온 '+c+' 신나게 읽어줘 .'+'\n')
            out.write('\n')
