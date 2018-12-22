from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtCore import Qt
import sys
from test_hog_svm import predict
from VGG16 import vggpredict

classes = {0:'plane', 1:'car', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

class Example(QMainWindow):

    def __init__(self):
        super().__init__()

        
        self.imagepath = './cifar-10/test/0/0_3.jpg'
        self.hogpath = 'hog.jpg'
        self.vggpath = 'vgg.jpg'
        self.initUI()


    def initUI(self):

        self.lb1 = QLabel(self)
        self.lb1.setGeometry(50,200,188,188)

        self.il = QLabel(self)
        self.il.setGeometry(90,400,100,30)
        self.il.setFont(QFont('Roman times', 20))
        self.il.setAlignment(Qt.AlignCenter)

        self.fl = QLabel(self)
        self.fl.setGeometry(485,10,100,30)
        self.fl.setFont(QFont('Roman times', 20))
        self.fl.setText('Features')
        self.fl.setAlignment(Qt.AlignCenter)

        self.pc = QLabel(self)
        self.pc.setGeometry(660,10,200,30)
        self.pc.setFont(QFont('Roman times', 20))
        self.pc.setText('Predict Class')
        self.pc.setAlignment(Qt.AlignCenter)

        self.lb2 = QLabel(self)
        self.lb2.setGeometry(460,50,156,156)

        self.cl2 = QLabel(self)
        self.cl2.setGeometry(660,110,200,30)
        self.cl2.setFont(QFont('Roman times', 20))
        self.cl2.setAlignment(Qt.AlignCenter)

        self.lb3 = QLabel(self)
        self.lb3.setGeometry(460,220,156,156)

        self.cl3 = QLabel(self)
        self.cl3.setGeometry(660,280,200,30)
        self.cl3.setFont(QFont('Roman times', 20))
        self.cl3.setAlignment(Qt.AlignCenter)

        btn1 = QPushButton("Open Image", self)
        btn1.move(90, 160)
        btn1.clicked.connect(self.button1Clicked)

        btn2 = QPushButton("Hog+SVM", self)
        btn2.move(310,120)
        btn2.clicked.connect(self.button2Clicked)

        btn3 = QPushButton("VGG16", self)
        btn3.move(310,280)
        btn3.clicked.connect(self.button3Clicked)

        
        self.setGeometry(100, 100, 900, 600)
        self.setWindowTitle('Image Classification')
        self.show()

        self.firstOpen()

    def button1Clicked(self):
        openfile_name = QFileDialog.getOpenFileName(self,'Choose Image','','Image files(*.jpg)')
        self.imagepath = openfile_name[0]
        self.il.setText(classes[int(self.imagepath.split('/')[-1].split('_')[0])])
        image = QPixmap(self.imagepath)
        self.lb1.setPixmap(image)
        self.lb1.setScaledContents(True)
    
    def button2Clicked(self):
        if self.imagepath == '':
            reply = QMessageBox.critical(self,'Error!','You should choose an image first!',QMessageBox.Yes | QMessageBox.No)
        else:
            hogsvm = predict(self.imagepath)
            image = QPixmap(self.hogpath)
            self.lb2.setPixmap(image)
            self.lb2.setScaledContents(True)
            self.cl2.setText(classes[hogsvm])
            
    def button3Clicked(self):
        if self.imagepath == '':
            reply = QMessageBox.critical(self,'Error!','You should choose an image first!',QMessageBox.Yes | QMessageBox.No)
        else:
            vgg = vggpredict(self.imagepath)
            image = QPixmap(self.vggpath)
            self.lb3.setPixmap(image)
            self.lb3.setScaledContents(True)
            self.cl3.setText(classes[int(vgg[0])])
    
    def firstOpen(self):
        image = QPixmap(self.imagepath)
        self.il.setText(classes[int(self.imagepath.split('/')[-1].split('_')[0])])
        self.lb1.setPixmap(image)
        self.lb1.setScaledContents(True)
        self.button2Clicked()
        self.button3Clicked()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())