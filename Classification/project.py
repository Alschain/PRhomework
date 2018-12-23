from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtCore import Qt
import sys
from test_hog_svm import hogsvmpredict
from test_lbp_knn import lbpknnpredict
from VGG16 import vggpredict

classes = {0:'plane', 1:'car', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

class Classification(QMainWindow):

    def __init__(self):
        super().__init__()

        self.imagepath = './cifar-10/test/9/9_28.jpg'
        self.hogpath = 'hog.jpg'
        self.vggpath = 'vgg.jpg'
        self.lbppath = 'lbp.jpg'
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
        self.pc.setText('Class Predicted')
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

        self.lb4 = QLabel(self)
        self.lb4.setGeometry(460,390,156,156)

        self.cl4 = QLabel(self)
        self.cl4.setGeometry(660,450,200,30)
        self.cl4.setFont(QFont('Roman times', 20))
        self.cl4.setAlignment(Qt.AlignCenter)

        self.lb5 = QLabel(self)
        self.lb5.setGeometry(600,575,300,20)
        self.lb5.setFont(QFont('Roman times', 13))
        self.lb5.setText('Build by Wang, Jiang and Yang. ')
        self.lb5.setAlignment(Qt.AlignRight)

        btn1 = QPushButton('Open Image', self)
        btn1.move(45, 160)
        btn1.clicked.connect(self.button1Clicked)

        btn5 = QPushButton('Run All', self)
        btn5.move(145,160)
        btn5.clicked.connect(self.button5Clicked)

        btn2 = QPushButton('Hog+SVM', self)
        btn2.move(310,120)
        btn2.clicked.connect(self.button2Clicked)

        btn3 = QPushButton('VGG16', self)
        btn3.move(310,280)
        btn3.clicked.connect(self.button3Clicked)

        btn4 = QPushButton('LBP+KNN',self)
        btn4.move(310,440)
        btn4.clicked.connect(self.button4Clicked)
        
        self.setGeometry(100, 100, 900, 600)
        self.setWindowTitle('Image Classification on Cifar-10')
        self.setWindowFlags(Qt.WindowMinimizeButtonHint)
        self.setFixedSize(self.width(), self.height())
        self.show()

        self.firstOpen()

    def button1Clicked(self):
        openfile_name = QFileDialog.getOpenFileName(self,'Choose Image','','Image files(*.jpg)')
        self.imagepath = openfile_name[0]
        if openfile_name[0] != '':
            self.il.setText(classes[int(self.imagepath.split('/')[-1].split('_')[0])])
            image = QPixmap(self.imagepath)
            self.lb1.setPixmap(image)
            self.lb1.setScaledContents(True)
            self.lb2.clear()
            self.cl2.clear()
            self.lb3.clear()
            self.cl3.clear()
            self.lb4.clear()
            self.cl4.clear()
    
    def button2Clicked(self):
        if self.imagepath == '':
            reply = QMessageBox.critical(self,'Error!','You should choose an image first!',QMessageBox.Yes | QMessageBox.No)
        else:
            hogsvm = hogsvmpredict(self.imagepath)
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
    
    def button4Clicked(self):
        if self.imagepath == '':
            reply = QMessageBox.critical(self,'Error!','You should choose an image first!',QMessageBox.Yes | QMessageBox.No)
        else:
            lbpknn = lbpknnpredict(self.imagepath)
            image = QPixmap(self.lbppath)
            self.lb4.setPixmap(image)
            self.lb4.setScaledContents(True)
            self.cl4.setText(classes[lbpknn])

    def button5Clicked(self):
        if self.imagepath == '':
            reply = QMessageBox.critical(self,'Error!','You should choose an image first!',QMessageBox.Yes | QMessageBox.No)
        else:
            self.button2Clicked()
            self.button3Clicked()
            self.button4Clicked()
    
    def firstOpen(self):
        image = QPixmap(self.imagepath)
        self.il.setText(classes[int(self.imagepath.split('/')[-1].split('_')[0])])
        self.lb1.setPixmap(image)
        self.lb1.setScaledContents(True)
        self.button2Clicked()
        self.button3Clicked()
        self.button4Clicked()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    clscifar = Classification()
    sys.exit(app.exec_())