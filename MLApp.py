import pickle

import customtkinter as tk
from tkinter import filedialog

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.tree import DecisionTreeClassifier

from ScrollableTable import ScrollableTable

DATASETS = ["BankSim.csv", "BAF-V I.csv", "BAF-V II.csv", "SyntheticDataset.csv"]
MODELS = [
    "AdaBoost",
    "Decision Tree",
    "Gaussian Naive Bayes",
    "K-Nearest Neighbors",
    "Logistic Regression",
    "Multilayer Perceptron",
    "Random Forest",
]
NORMALIZATION = ["Not Normalized", "Normalizer()", "MinMaxScaler()"]


def create_donuts(title, row, column):
    metric_fig, metric_ax = plt.subplots(figsize=(1.7, 1.7), facecolor='#242424')
    metric_ax.pie([0, 1], labels=["", ""], colors=['#1f77b4', '#e7e7e7'], startangle=90, counterclock=False,
                  wedgeprops=dict(width=0.3))
    metric_ax.text(0, 0, "0.0%", color='white', ha='center', va='center', fontsize=14)
    metric_ax.set_title(title, fontsize=11, color='white')

    metric_ax.axis('equal')
    metric_canvas = FigureCanvasTkAgg(metric_fig, master=root)
    metric_canvas.get_tk_widget().grid(row=row, column=column, rowspan=3, padx=10, pady=5)
    return metric_fig, metric_ax


def plot_donut_chart(value, ax, fig, title):
    ax.clear()
    ax.pie([value, 100 - value], labels=["", ""], colors=['#1f77b4', '#e7e7e7'], startangle=90, counterclock=False,
           wedgeprops=dict(width=0.3))
    ax.text(0, 0, f"{value:.2f}%", color='white', ha='center', va='center', fontsize=14)
    ax.set_title(title, color='white')
    fig.canvas.draw()


class MLApp:
    def __init__(self, root):
        self.table_frame = None
        tk.set_appearance_mode("Dark")
        self.y_test = None
        self.y_train = None
        self.x_test_norm = None
        self.x_train_norm = None
        self.x_test = None
        self.x_train = None
        self.x = None
        self.y = None
        self.dataset = None

        self.model = None

        self.model_directory = "C:\\Users\\BOGI\\Desktop\\MLApp\\TrainedModels"
        self.model_pkl = None

        self.root = root
        self.root.title("Machine Learning Application")

        self.dataset_label = tk.CTkLabel(root, text="Dataset: ")
        self.dataset_label.grid(row=0, column=0, padx=10, pady=5)
        self.dataset_button = tk.CTkButton(
            root, text="Browse", command=self.browse_dataset
        )
        self.dataset_button.grid(row=0, column=1, padx=10, pady=5)

        self.model_label = tk.CTkLabel(root, text="Model:")
        self.model_label.grid(row=1, column=0, padx=10, pady=5)

        self.norm_label = tk.CTkLabel(root, text="Normalization Method:")
        self.norm_label.grid(row=2, column=0, padx=10, pady=5)

        self.norm_var = tk.StringVar(root)
        self.norm_var.set(NORMALIZATION[0])
        self.norm_dropdown = tk.CTkOptionMenu(root, values=NORMALIZATION, command=self.normalize,
                                              variable=self.norm_var)
        self.norm_dropdown.grid(row=2, column=1, padx=10, pady=5)

        self.epoch_label = tk.CTkLabel(root, text="Number of Epochs:")
        self.epoch_label.grid(row=3, column=0, padx=10, pady=5)

        self.epoch_box = tk.CTkTextbox(root)
        self.epoch_box.grid(row=3, column=1, padx=10, pady=5)
        self.epoch_box.configure(height=20, width=150)
        self.epoch_box.insert("0.0", "200")

        self.tpr_fig, self.tpr_ax = create_donuts("TPR", 0, 3)
        self.fpr_fig, self.fpr_ax = create_donuts("FPR", 0, 4)
        self.accuracy_fig, self.accuracy_ax = create_donuts("Accuracy", 3, 3)
        self.precision_fig, self.precision_ax = create_donuts("Precision", 3, 4)
        self.tnr_fig, self.tnr_ax = create_donuts("TNR", 0, 5)
        self.fnr_fig, self.fnr_ax = create_donuts("FNR", 0, 6)
        self.f_score_fig, self.f_score_ax = create_donuts("F1-score", 3, 5)
        self.b_accuracy_fig, self.b_accuracy_ax = create_donuts("Balanced Accuracy", 3, 6)

        self.model_var = tk.StringVar(root)
        self.model_var.set(MODELS[0])
        self.model_dropdown = tk.CTkOptionMenu(
            root,
            values=MODELS,
        )
        self.model_dropdown.grid(row=1, column=1, padx=10, pady=5)

        self.train_button = tk.CTkButton(
            root, text="Train Model", command=self.train_model
        )

        self.load_button = tk.CTkButton(
            root, text="Load Model", command=self.load_model
        )

        self.train_button.grid(row=4, column=0, padx=10, pady=5)
        self.load_button.grid(row=4, column=1, padx=10, pady=5)

        self.fig, self.ax = plt.subplots(facecolor="#242424")
        self.ax.set_facecolor("#242424")
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        plt.gca().spines['bottom'].set_color('white')
        plt.gca().spines['left'].set_color('white')
        plt.tick_params(axis='x', colors='white')
        plt.tick_params(axis='y', colors='white')
        self.canvas.get_tk_widget().grid(row=6, column=3, columnspan=4, rowspan=8, padx=10, pady=5)

    def browse_dataset(self):
        file_path = filedialog.askopenfilename(title="Select Dataset File")
        filename = os.path.basename(file_path)
        if filename:
            self.dataset = pd.read_csv(file_path)
            if filename == DATASETS[0]:
                self.update_model_dropdown(
                    MODELS
                )
            elif filename in [DATASETS[1], DATASETS[2]]:
                self.update_model_dropdown(
                    [
                        MODELS[0], MODELS[1], MODELS[2], MODELS[3]
                    ]
                )
            elif filename == DATASETS[3]:
                self.update_model_dropdown(
                    [
                        MODELS[0], MODELS[1], MODELS[2], MODELS[5], MODELS[6]
                    ]
                )
            self.table_frame = ScrollableTable(root, file_path)
            self.table_frame.grid(row=6, column=0, columnspan=2, rowspan=8, padx=10, pady=5)

            self.transform_categorical(filename)

            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x, self.y, test_size=0.2, random_state=100
            )

    def update_model_dropdown(self, model_list):
        self.model_dropdown.configure(values=model_list)

    def train_model(self):
        # model = None
        model_type = self.model_dropdown.get()
        if model_type == MODELS[0]:
            self.model = AdaBoostClassifier(random_state=42)
        elif model_type == MODELS[1]:
            self.model = DecisionTreeClassifier()
        elif model_type == MODELS[2]:
            self.model = GaussianNB()
        elif model_type == MODELS[3]:
            self.model = KNeighborsClassifier(n_neighbors=3)
        elif model_type == MODELS[4]:
            self.model = LogisticRegression(solver="lbfgs", max_iter=1000)
        elif model_type == MODELS[5]:
            epochs = self.epoch_box.get("0.0", "end")
            self.model = MLPClassifier(
                hidden_layer_sizes=(6, 5), random_state=5, learning_rate_init=0.01, max_iter=int(epochs)
            )
        elif model_type == MODELS[6]:
            self.model = RandomForestClassifier(
                criterion="gini", max_depth=8, min_samples_split=10, random_state=5
            )
        self.normalize(self.norm_dropdown.get())
        self.model.fit(self.x_train_norm, self.y_train)
        y_pred = self.model.predict(self.x_test_norm)

        self.calculate_scores(y_pred)

        model_filename = model_type + "_" + self.norm_dropdown.get() + ".pkl"
        model_pkl_file = os.path.join(self.model_directory, model_filename)

        os.makedirs(self.model_directory, exist_ok=True)
        with open(model_pkl_file, "wb") as file:
            pickle.dump(self.model, file)
            print("Created")

    def calculate_scores(self, y_pred):
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Accuracy:", accuracy)
        plot_donut_chart(accuracy * 100, self.accuracy_ax, self.accuracy_fig, "Accuracy")

        tpr = recall_score(self.y_test, y_pred)
        plot_donut_chart(tpr * 100, self.tpr_ax, self.tpr_fig, "TPR")
        print("TPR:", tpr)

        tnr = recall_score(self.y_test, y_pred, pos_label=0)
        plot_donut_chart(tnr * 100, self.tnr_ax, self.tnr_fig, "TNR")
        print("TNR:", tnr)

        fpr = 1 - recall_score(self.y_test, y_pred)
        plot_donut_chart(fpr * 100, self.fpr_ax, self.fpr_fig, "FPR")
        print("FPR:", fpr)

        fnr = 1 - recall_score(self.y_test, y_pred, pos_label=0)
        plot_donut_chart(fnr * 100, self.fnr_ax, self.fnr_fig, "FNR")
        print("FNR:", fnr)

        precision = precision_score(self.y_test, y_pred)
        plot_donut_chart(precision * 100, self.precision_ax, self.precision_fig, "Precision")
        print("Precision:", precision)

        f1 = f1_score(self.y_test, y_pred)
        plot_donut_chart(f1 * 100, self.f_score_ax, self.f_score_fig, "F1-score")
        print("F1 score:", f1)

        ba = balanced_accuracy_score(self.y_test, y_pred)
        plot_donut_chart(ba * 100, self.b_accuracy_ax, self.b_accuracy_fig, "Balanced accuracy")
        print("Balanced accuracy:", ba)

        self.plot_learning_curve(self.model)

    def load_model(self):
        model_filename = self.model_dropdown.get() + "_" + self.norm_dropdown.get() + ".pkl"
        model_pkl_file = os.path.join(self.model_directory, model_filename)

        with open(model_pkl_file, "rb") as file:
            self.model = pickle.load(file)

        y_pred = self.model.predict(self.x_test)

        self.calculate_scores(y_pred)

    def normalize(self, choice):
        if choice == "Not Normalized":
            self.x_train_norm = self.x_train
            self.x_test_norm = self.x_test

            self.table_frame.populate_table()
        elif choice == "Normalizer()":
            self.x_train_norm = Normalizer().fit_transform(self.x_train)
            self.x_test_norm = Normalizer().fit_transform(self.x_test)

            self.table_frame.update_data(self.x_train_norm)
        elif choice == "MinMaxScaler()":
            self.x_train_norm = MinMaxScaler().fit_transform(self.x_train)
            self.x_test_norm = MinMaxScaler().fit_transform(self.x_test)

            self.table_frame.update_data(self.x_train_norm)

    def transform_categorical(self, filename):
        if filename == DATASETS[0]:
            self.dataset.customer = pd.Categorical(self.dataset.customer)
            self.dataset["customer"] = self.dataset.customer.cat.codes

            self.dataset.age = pd.Categorical(self.dataset.age)
            self.dataset["age"] = self.dataset.age.cat.codes

            self.dataset.gender = pd.Categorical(self.dataset.gender)
            self.dataset["gender"] = self.dataset.gender.cat.codes

            self.dataset.zipcodeOri = pd.Categorical(self.dataset.zipcodeOri)
            self.dataset["zipcodeOri"] = self.dataset.zipcodeOri.cat.codes

            self.dataset.merchant = pd.Categorical(self.dataset.merchant)
            self.dataset["merchant"] = self.dataset.merchant.cat.codes

            self.dataset.zipMerchant = pd.Categorical(self.dataset.zipMerchant)
            self.dataset["zipMerchant"] = self.dataset.zipMerchant.cat.codes

            self.dataset.category = pd.Categorical(self.dataset.category)
            self.dataset["category"] = self.dataset.category.cat.codes

            self.y = self.dataset["fraud"]
            self.x = self.dataset.drop("fraud", axis=1)
        elif filename in [DATASETS[1], DATASETS[2]]:
            self.dataset.payment_type = pd.Categorical(self.dataset.payment_type)
            self.dataset["payment_type"] = self.dataset.payment_type.cat.codes

            self.dataset.employment_status = pd.Categorical(
                self.dataset.employment_status
            )
            self.dataset["employment_status"] = self.dataset.employment_status.cat.codes

            self.dataset.housing_status = pd.Categorical(self.dataset.housing_status)
            self.dataset["housing_status"] = self.dataset.housing_status.cat.codes

            self.dataset.source = pd.Categorical(self.dataset.source)
            self.dataset["source"] = self.dataset.source.cat.codes

            self.dataset.device_os = pd.Categorical(self.dataset.device_os)
            self.dataset["device_os"] = self.dataset.device_os.cat.codes

            self.y = self.dataset["fraud_bool"]

            self.x = self.dataset.drop("fraud_bool", axis=1)
        elif filename == DATASETS[3]:
            self.dataset.type = pd.Categorical(self.dataset.type)
            self.dataset["type"] = self.dataset.type.cat.codes

            self.dataset.branch = pd.Categorical(self.dataset.branch)
            self.dataset["branch"] = self.dataset.branch.cat.codes

            self.dataset.nameOrig = pd.Categorical(self.dataset.nameOrig)
            self.dataset["nameOrig"] = self.dataset.nameOrig.cat.codes

            self.dataset.nameDest = pd.Categorical(self.dataset.nameDest)
            self.dataset["nameDest"] = self.dataset.nameDest.cat.codes

            self.dataset.AcctType = pd.Categorical(self.dataset.AcctType)
            self.dataset["AcctType"] = self.dataset.AcctType.cat.codes

            self.dataset.DateOfTransaction = pd.Categorical(
                self.dataset.DateOfTransaction
            )
            self.dataset["DateOfTransaction"] = self.dataset.DateOfTransaction.cat.codes

            self.dataset.TimeOfDay = pd.Categorical(self.dataset.TimeOfDay)
            self.dataset["TimeOfDay"] = self.dataset.TimeOfDay.cat.codes

            self.y = self.dataset["isFraud"]
            self.x = self.dataset.drop("isFraud", axis=1)

    def plot_learning_curve(self, model):
        train_sizes, train_scores, test_scores = learning_curve(
            model, self.x, self.y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        self.ax.clear()
        self.ax.set_title("Learning Curve", color="white")
        self.ax.set_xlabel("Training examples", color="white")
        self.ax.set_ylabel("Score", color="white")

        self.ax.grid()

        self.ax.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="#eb7a34",
        )
        self.ax.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="#0697c7",
        )

        self.ax.plot(
            train_sizes,
            train_scores_mean,
            'o-', color="#eb7a34", label="Training score"
        )
        self.ax.plot(
            train_sizes,
            test_scores_mean,
            'o-', color="#0697c7", label="Cross-validation score"
        )

        self.ax.legend(loc="lower right")
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.CTk()
    app = MLApp(root)

    for i in range(0, 6):
        root.columnconfigure(i, weight=1)

    root.mainloop()
    
