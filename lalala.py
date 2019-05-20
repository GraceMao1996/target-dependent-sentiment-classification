from Tkinter import *
import tkMessageBox

class Application(Frame):
	def __init__(self,master=None):
		Frame.__init__(self, master)
		self.createWidgets()

	def createWidgets(self):
		self.nameInput = Entry(self)
		self.nameInput.pack()
		self.alertButton = Button(self, text='Hello',command = self.hello)
		self.alertButton.pack()

	def hello(self):
		name = self.nameInput.get() or 'World'
		tkMassegeBox.showInfo('Message','Hello, %s' % name)

app = Application()
app.master.title('hello, world!')
app.mainloop()
