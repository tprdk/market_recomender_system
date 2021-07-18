import itertools
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from PyQt5.uic import loadUi
from PyQt5.QtCore import QAbstractTableModel, Qt, QObject, pyqtSignal, pyqtSlot, QThread
import sys

from generate_mock_data import read_data_frame
from user_based_cf import create_user_item_matrix, predict_selected_customer, \
    calculate_zero_centered_matrix, predict_next_market
from dfm import *


class pandasModel(QAbstractTableModel):
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[section]
        return None

class Worker(QObject):
    finished = pyqtSignal()
    intReady = pyqtSignal(int)
    ui = None

    @pyqtSlot()
    def predict_selected(self):
        self.ui.label_accuracy.setText('')
        self.ui.label_target.setText('')
        self.ui.listWidget_predicted.clear()
        customer_index = self.ui.tableView_customers.selectionModel().selectedRows()[0].row()
        df = self.ui.df
        self.intReady.emit(0)
        self.intReady.emit(15)
        if self.ui.comboBox_model.currentText() == 'Collaborative Filtering':
            print('Collaborative Filtering')
            matrix, test_market = create_user_item_matrix(df, 75)
            self.intReady.emit(15)
            target, market_recommendations = \
                predict_selected_customer(matrix, df.loc[[customer_index]], customer_index,
                                          int(self.ui.comboBox_similar_user_count.currentText()),
                                          int(self.ui.comboBox_recommendation_count.currentText()))
            self.intReady.emit(100)
            self.ui.label_target.setText(str(target))
            for element in market_recommendations[0]:
                self.ui.listWidget_predicted.addItem(str(element))
            if target in market_recommendations:
                self.ui.label_accuracy.setText('Başarılı')
            else:
                self.ui.label_accuracy.setText('Başarısız')
        else:
            if self.ui.pretrained_model_path != '':
                print('Deep Factorization Machine')
                recommended_market_count = int(self.ui.comboBox_recommendation_count.currentText())
                self.intReady.emit(15)
                self.ui.dfm_model.eval()
                selected_customer = next(itertools.islice(self.ui.test_data_loader, customer_index, None))
                with torch.no_grad():
                    fields, target, visited = selected_customer
                    # load features and targets to gpu
                    fields, target = fields.to(self.ui.device), target.to(self.ui.device)
                    # predict
                    y = self.ui.dfm_model(fields)
                    # get predictions to cpu
                    targets = np.array(target.to('cpu'))
                    predictions = np.array(y.to('cpu'))
                    visited_market = np.array(visited.to('cpu'))
                    # create mask array and give 0 score for visited markets
                    mask = np.ones(shape=(75, 1), dtype='int')
                    for visit in visited_market[0]:
                        if visit != -1:
                            mask[visit] = 0
                    # mask visited markets scores
                    predictions = predictions * mask.reshape(1, -1)
                    # get top n scored market for prediction
                    max_rated = np.argsort(-predictions)[0][:recommended_market_count]
                    # if top scored markets contains target market this means we got hit
                    self.intReady.emit(100)
                    if targets[0] in max_rated:
                        self.ui.label_accuracy.setText('Başarılı')
                    else:
                        self.ui.label_accuracy.setText('Başarısız')
                    self.ui.label_target.setText(str(targets[0]))
                    for element in max_rated:
                        self.ui.listWidget_predicted.addItem(str(element))
        self.finished.emit()


    @pyqtSlot()
    def predict_all(self):  # A slot takes no params
        self.intReady.emit(0)
        self.intReady.emit(5)
        if self.ui.comboBox_model.currentText() == 'Collaborative Filtering':
            print('Collaborative Filtering')
            df = self.ui.df
            df.reset_index(drop=True, inplace=True)
            sample_count = len(df)
            matrix, test_market = create_user_item_matrix(df, 75)
            self.intReady.emit(15)
            # cf
            zero_centered_matrix = calculate_zero_centered_matrix(matrix)
            acc = 0
            increase_factor = int(sample_count / 80)
            percent = increase_factor
            for index, row in enumerate(zero_centered_matrix):
                temp_rate = row[test_market[index]]
                row[test_market[index]] = 0.
                # predict_rate
                if index % 100 == 0:
                    print(f'index : {index} - acc : {acc}')
                if test_market[index] in predict_next_market(zero_centered_matrix, matrix, index, test_market[index],
                                                             int(self.ui.comboBox_similar_user_count.currentText()),
                                                             int(self.ui.comboBox_recommendation_count.currentText())):
                    acc += 1
                row[test_market[index]] = temp_rate
                if percent == index:
                    self.intReady.emit(1)
                    percent += increase_factor
                self.ui.label_accuracy.setText(f'%{acc * 100 / (index + 1)}')
            self.ui.label_accuracy.setText(f'%{acc * 100 / sample_count}')

        elif self.ui.comboBox_model.currentText() == 'Deep Factorization Machine':
            print('Deep Factorization Machine')
            recommended_market_count = int(self.ui.comboBox_recommendation_count.currentText())
            self.intReady.emit(15)
            increase_factor = int(len(self.ui.test_data_loader) / 80)
            percent = increase_factor

            if self.ui.pretrained_model_path is not None:
                self.ui.dfm_model.eval()
                acc = 0
                sample = 0
                with torch.no_grad():
                    for fields, target, visited in tqdm.tqdm(self.ui.test_data_loader, smoothing=0, mininterval=1.0):
                        # load features and targets to gpu
                        fields, target = fields.to(self.ui.device), target.to(self.ui.device)
                        # predict
                        y = self.ui.dfm_model(fields)
                        # get predictions to cpu
                        targets = np.array(target.to('cpu'))
                        predictions = np.array(y.to('cpu'))
                        visited_market = np.array(visited.to('cpu'))

                        # create mask array and give 0 score for visited markets
                        mask = np.ones(shape=(75, 1), dtype='int')
                        for visit in visited_market[0]:
                            if visit != -1:
                                mask[visit] = 0

                        # mask visited markets scores
                        predictions = predictions * mask.reshape(1, -1)
                        # get top n scored market for prediction
                        max_rated = np.argsort(-predictions)[0][:recommended_market_count]
                        # if top scored markets contains target market this means we got hit
                        if targets[0] in max_rated:
                            acc += 1
                        sample += 1
                        if percent == sample:
                            self.intReady.emit(1)
                            percent += increase_factor
                        self.ui.label_accuracy.setText(f'%{acc * 100 / (sample + 1)}')
                    self.ui.label_accuracy.setText(f'%{acc * 100 / sample}')
        self.intReady.emit(100)
        self.finished.emit()


class Recommender_Window(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi('../ui/ui.ui', self)
        self.setWindowIcon(QtGui.QIcon('../ui/icon.png'))
        self.model = QtGui.QStandardItemModel(self)
        self.df = None
        self.param = None

        self.device = 'cuda:0'
        self.pretrained_model_path = ''
        self.train_dataset_path = ''
        self.test_dataset_path = ''
        self.dataset_test = None
        self.dataset_train = None
        self.test_data_loader = None
        self.dfm_model = None

        self.maxVal = 0

        self.comboBox_model.addItems(['Collaborative Filtering', 'Deep Factorization Machine'])
        self.comboBox_model.currentTextChanged.connect(self.model_selection_changed)

        self.comboBox_similar_user_count.addItems(['50', '75', '100'])
        self.comboBox_recommendation_count.addItems(['10', '15', '20'])

        self.pushButton_file_path.clicked.connect(self.open_file_dialog_for_data)
        self.pushButton_predict_all.clicked.connect(self.predict_all)
        self.pushButton_predict_selected.clicked.connect(self.predict_selected)
        self.pushButton_model_path.clicked.connect(self.open_file_dialog_for_model)

        self.label_model_path.setVisible(False)
        self.pushButton_model_path.setVisible(False)


    def start_worker(self):
        self.worker, self.thread = Worker(), QThread()
        self.worker.ui = self
        self.worker.intReady.connect(self.update_progress_bar)
        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.thread.quit)

        if self.param == 'predict_all':
            self.thread.started.connect(self.worker.predict_all)
            self.thread.start()
        elif self.param == 'predict_selected':
            self.thread.started.connect(self.worker.predict_selected)
            self.thread.start()


    def model_selection_changed(self):
        if self.comboBox_model.currentText() == 'Collaborative Filtering':
            self.comboBox_similar_user_count.setVisible(True)
            self.label_similar_user_count.setVisible(True)
            self.label_model_path.setVisible(False)
            self.pushButton_model_path.setVisible(False)
        else:
            self.comboBox_similar_user_count.setVisible(False)
            self.label_similar_user_count.setVisible(False)
            self.label_model_path.setVisible(True)
            self.pushButton_model_path.setVisible(True)


    def open_file_dialog_for_data(self):
        fileName, _ = QFileDialog.getOpenFileName(None, "Select Dataset", "../", "CSV Files (*.csv);")
        print(f'file : {fileName}')
        if fileName != '':
            print(f'file :: {fileName}')
            self.label_file_path.setText(fileName)
            self.load_csv()


    def load_model(self):
        self.train_dataset_path = self.label_file_path.text().replace('.csv', '_train.dat')
        self.test_dataset_path = self.label_file_path.text().replace('.csv', '_test.dat')

        self.dataset_train = get_dataset('Market_User', self.train_dataset_path, train=True)
        self.dataset_test = get_dataset('Market_User', self.test_dataset_path, train=False)
        self.test_data_loader = DataLoader(self.dataset_test, batch_size=75)

        self.dfm_model = get_model('dfm', self.dataset_train).to(self.device)
        self.dfm_model.load_state_dict(torch.load(self.pretrained_model_path))


    def open_file_dialog_for_model(self):
        fileName, _ = QFileDialog.getOpenFileName(None, "Select Dataset", "../",
                                                  "Pretrained Files (*.pt);")
        if fileName != '':
            self.label_model_path.setText(fileName)
            self.pretrained_model_path = fileName
            self.load_model()

    def update_progress_bar(self, value):
        if self.progressBar_predict.value() + value < 100:
            self.progressBar_predict.setValue(self.progressBar_predict.value() + value)
        elif value == 0:
            self.progressBar_predict.setValue(0)
        else:
            self.progressBar_predict.setValue(100)

    def load_csv(self):
        self.df = read_data_frame(self.label_file_path.text())
        model = pandasModel(self.df)
        self.tableView_customers.setModel(model)

    def predict_all(self):
        self.param = 'predict_all'
        self.start_worker()

    def predict_selected(self):
        self.param = 'predict_selected'
        self.start_worker()


if __name__ == "__main__":
    app = QApplication([])
    window = Recommender_Window()
    window.show()
    sys.exit(app.exec_())
