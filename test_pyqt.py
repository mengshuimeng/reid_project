# test_pyqt.py
import sys
print("python:", sys.executable)
try:
    from PyQt5 import QtWidgets
    print("import PyQt5 OK")
    app = QtWidgets.QApplication([])
    w = QtWidgets.QWidget()
    w.setWindowTitle("Qt test")
    w.resize(200,100)
    w.show()
    print("QApplication created, starting exec...")
    app.processEvents()
    print("processEvents OK, exiting.")
except Exception as e:
    print("PyQt5 exception:", e)
