"""
BFM-Report class for creation of an automatically gernerated PDF protocol
Author: Michael Kranz, M.Sc.
Date: 04.04.2023
Version: V1.0
"""

# Import
from fpdf import FPDF # pip install fpdf
import matplotlib.pyplot as plt

class BFMReport(FPDF):
    def header(self):
        self.image('OTH_Logo_BFM.png', 10, 8, 25)
        self.image('qr_code.png', 170, 6, 30)
        # font
        self.set_font('helvetica', 'B', 20)
        # title
        self.cell(0, 10, 'Report', border=False, ln=True, align='C')
        self.ln(5)
        self.ln(20)

    def footer(self):
        # position
        self.set_y(-15)
        # font
        self.set_font('helvetica', 'I', 10)
        # page number
        self.cell(0, 10, f'(c) Michael Kranz and Daniel Pointner \t Page {self.page_no()}/{{nb}}', align='R')

    def chapter_title(self, ch_num, ch_title, link):
        self.set_link(link)
        self.set_font('helvetica', '', 20)
        chapter_title = f'{ch_num}. {ch_title}'
        self.cell(0, 5, chapter_title, ln=True)
        self.ln()

    def section(self, ch_num, sec_num, sec_title, link):
        self.set_link(link)
        self.set_font('helvetica', '', 16)
        section_title = f'{ch_num}.{sec_num} {sec_title}'
        self.cell(0, 5, section_title, ln=True)
        # def chapter_body(self):
        # self.set_font('helvetica', '', 12)
        # self.cell(0,5, 'Lorem Ipsum iasdoiieu gauefhu  gaefu uageufhu  gauefiugo gsef', ln=True)
        self.ln()

    def information(self, day, script_version, experiement_number, log_number):#, data_csv):
        self.set_font('helvetica', '', 12)
        self.ln(40)
        self.cell(0, 10, f'Created with version:\t {script_version}', ln=True, align='L')
        self.cell(0, 10, f'Date:\t {day}', ln=True, align='L')
        self.cell(0, 10, f'Experiment number:\t {experiement_number}', ln=True, align='L')
        self.cell(0, 10, f'Log number:\t {log_number}', ln=True, align='L')
        #self.cell(0, 10, f'Filename data: {data_csv}', ln=True, align='L')
        self.ln()

    def graph(self, time, data, title, name_graph, xlabel, ylabel, xcoord, ycoord):
        time = time.values.tolist()  # converting DataFrames into list (x coordinate)
        data = data.values.tolist()  # y-coordinate
        plt.figure()
        plt.plot(time, data, label=name_graph)
        plt.legend()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.savefig(f'{name_graph}.png', dpi=200)
        if ycoord == 0:
            self.image(f'{name_graph}.png', xcoord * 0.5 * self.w + 10, ycoord * 0.5 * self.w + 70, 0.4 * self.w)
        elif ycoord > 0:
            self.image(f'{name_graph}.png', xcoord * 0.5 * self.w + 10, ycoord * 0.5 * self.w + 20, 0.4 * self.w)

    def settings_report(self, data):
        line_height = self.font_size*2
        col_width = self.w / 6
        self.set_font('helvetica', '', 12)
        self.cell(0, 10, 'Table 1: Used settings for processing', ln=True, align='L')
        for row in data:
            self.multi_cell(col_width, line_height, row[0], border=1, ln=3, align='L')
            self.multi_cell(col_width*5, line_height, row[1], border=1, ln=3, align='L')
            self.ln(line_height)

    def print_chapter(self, ch_num, ch_title, link):
        self.add_page()
        self.chapter_title(ch_num, ch_title, link)
