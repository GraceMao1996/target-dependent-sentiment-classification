import re
from tkinter import *

 
class Application(Frame):
  def __init__(self, master=None):
    Frame.__init__(self, master)
    self.pack()
    self.createWidgets()
 
  def createWidgets(self):
    self.file_name = Entry(self)
    self.file_name.pack()
	self.target_words = Entry(self)
	self.target_words.pack()
    self.alertButton = Button(self, text='run', command=self.run)
    self.alertButton.pack()
 
  def run(self):
    file_name = self.file_name.get() 
	target_words = self.target_words.get()
	_WORD_SPLIT = re.compile("([.,!?\"':;)(])")

	file = open(file_name,'r')
	passage = file.read().strip().lower()
	passage = _WORD_SPLIT.split(passage)

	sen = []

	f = open('data/classify.cor','w')
	for sentence in passage:
		if target_words in sentence:
		
		
			temp = re.compile(target_words)
			sentence = temp.sub('$T$', sentence)
			f.write(sentence)
			f.write('\n')
			f.write(target_words)
			f.write('\n')
			sen = sentence.split(' ')
	

	file.close()
	f.close()
    

app = Application()
# 设置窗口标题:
app.master.title('TD classification')
# 主消息循环:
app.mainloop()

