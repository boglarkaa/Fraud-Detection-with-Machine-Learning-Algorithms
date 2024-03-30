import csv
from tkinter import ttk
import tkinter as tk


class ScrollableTable(tk.Frame):
    def __init__(self, parent, data):
        super().__init__(parent)
        self.parent = parent
        self.data = data
        self.headers = None

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Treeview', background='#242424', foreground='white', rowheight=35, fieldbackground='#242424',
                        font=('Arial', 12))
        style.configure('Treeview.Heading', background='#242424', foreground='white', rowheight=35,
                        fieldbackground='#242424', font=('Arial', 12))

        self.table_frame = tk.Frame(self, width=630, height=450)
        self.table_frame.pack_propagate(False)
        self.table_frame.pack(side="top", fill="both", expand=True)

        self.x_scrollbar = ttk.Scrollbar(self, orient="horizontal")

        self.table = ttk.Treeview(self.table_frame, show="headings", xscrollcommand=self.x_scrollbar.set)

        self.x_scrollbar.config(command=self.table.xview)

        self.x_scrollbar.pack(side="bottom", fill="x")
        self.table.pack(side="left", fill="both", expand=True)

        #  self.table.pack(side="left", fill="both", expand=True)

        self.populate_table()

    def populate_table(self):
        self.table.delete(*self.table.get_children())

        with open(self.data, newline="") as csvfile:
            csvreader = csv.reader(csvfile)
            self.headers = next(csvreader)
            self.table["columns"] = self.headers
            self.table.heading("#0", text="#")
            for header in self.headers:
                self.table.heading(header, text=header)
                self.table.column(header, anchor=tk.CENTER, width=70, minwidth=70)
            for i, row in enumerate(csvreader):
                self.table.insert("", "end", iid=i, values=row)

    def update_data(self, data):
        self.table.delete(*self.table.get_children())
        for i, row in enumerate(data):
            formatted_row = ['{:.4f}'.format(col) for col in row]
            self.table.insert("", "end", iid=i, values=(i,) + tuple(formatted_row))
