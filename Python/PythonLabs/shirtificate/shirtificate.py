import fpdf
from fpdf import Align

def main():
    name = student()
    certificate(name)
    return

def student():
    name = input("ENTER YOUR NAME: ")
    return name.title()

def certificate(name):
    pdf = fpdf.FPDF(orientation='portrait', format = 'A4')
    pdf.set_font('Helvetica', size = 55)
    pdf.add_page()
    pdf.image('image.png', w = 190, h = 190, x = 10, y = 70 )
    pdf.text(33, 40, text = 'CS50 Shirtificate')
    pdf.set_font('Helvetica', size = 28)
    pdf.set_text_color(255, 255, 255)
    pdf.text(73, 140, text = f'{name} took CS50')
    pdf.output(name = 'doc.pdf')

if __name__ == '__main__':
    main()