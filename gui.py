import matplotlib
import os
import csv
import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure, Artist

matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.size'] = 10
matplotlib.use('Qt5Agg')

def saveCSV(array1,array2,filename):
    matrix = np.vstack((array1,array2))
    matrix = np.transpose(matrix)
    filename_mod = filename[:-7] + 'Dec' + filename[-7:]
    with open(filename_mod, 'w') as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow(r) for r in matrix]





class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=7, height =5, dpi=100, tight_layout = False):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.subplots_adjust(hspace=.5)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)

        FigureCanvas.__init__(self,self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot_filename(self, filename):
        route,file = os.path.split(filename)
        self.fig.suptitle(file)

    def plot_top(self, xAxis, yAxis, xLabel, yLabel, ref):
        self.ax1.plot(xAxis, yAxis)
        self.ax1.axis('on')
        self.ax1.grid('on')
        self.ax1.set_xlim(left=min(ref),right = max(ref))
        self.ax1.set_xlabel(xLabel)
        self.ax1.set_ylabel(yLabel)
        self.draw()

    def plot_bottom(self, xAxis, yAxis, xLabel, yLabel, ref):
        self.ax2.plot(xAxis, yAxis)
        self.ax2.axis('on')
        self.ax2.grid('on')
        self.ax2.set_xlim(left=min(ref),right = max(ref))
        self.ax2.set_xlabel(xLabel)
        self.ax2.set_ylabel(yLabel)
        self.draw()

    def clear_top(self):
        self.ax1.clear()
        self.draw()

    def clear_bottom(self):
        self.ax2.clear()
        self.draw()

class DSButton(QtWidgets.QLabel):

    def __init__(self, name, layout, parent, default=0, deci=4, maximum = 100, minimum = 0):
        
        self.label = QtWidgets.QLabel(name, parent)
        self.input = QtWidgets.QDoubleSpinBox(parent, decimals = deci)
        self.input.setRange(minimum, maximum)
        self.input.setSingleStep(1E-4)
        self.input.setValue(default)
        layout.addWidget(self.label)
        layout.addWidget(self.input)

    def get_value(self, factor):
        return self.input.value()/factor

class Tab(QtWidgets.QWidget):

    def __init__(self, name, parent):
        super(Tab, self).__init__()
        parent.addTab(self,name)
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setSpacing(10)
        self.layout.addStretch(0)
        self.setLayout(self.layout)