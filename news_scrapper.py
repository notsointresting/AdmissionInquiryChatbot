import requests
from bs4 import BeautifulSoup
from fpdf import FPDF
import os

# Font fallback class
class SmartPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.default_font = "Arial"
        self.unicode_font = "Noto"
        self.font_path = "fonts/NotoSansDevanagari-Regular.ttf"
        self.add_font(self.unicode_font, "", self.font_path, uni=True)
        self.set_font(self.default_font, size=12)

    def safe_cell(self, w, h, text, *args, **kwargs):
        try:
            self.set_font(self.default_font, size=12)
            self.cell(w, h, text, *args, **kwargs)
        except Exception as e:
            print(f"[⚠️ Unicode fallback] '{text}' failed with default font. Switching to Unicode.\n→ Error: {e}")
            self.set_font(self.unicode_font, size=12)
            self.cell(w, h, text, *args, **kwargs)

def get_latest_news(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        news_articles = soup.find_all('article', class_='exad-post-grid-three exad-col')

        news_list = []
        for article in news_articles[:5]:
            title = article.find('h3').find('a').text.strip()
            link = article.find('h3').find('a')['href']
            date = article.find('li', class_='exad-post-date').find('a').text.strip()

            news_list.append({'title': title, 'link': link, 'date': date})
        return news_list

    except Exception as e:
        print(f"[❌ ERROR] Failed to scrape {url}: {e}")
        return []

def create_pdf(all_news_dict, filename):
    pdf = SmartPDF()
    pdf.add_page()

    for source, news_list in all_news_dict.items():
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, f"Latest News from {source}", 0, 1, 'C')

        pdf.set_font('Arial', '', 12)
        for news in news_list:
            pdf.safe_cell(0, 5, f"Title: {news['title']}", 0, 1)
            pdf.safe_cell(0, 5, f"Link: {news['link']}", 0, 1)
            pdf.safe_cell(0, 5, f"Date: {news['date']}", 0, 1)
            pdf.cell(0, 8, '', 0, 1)  # Spacing between news items

        pdf.cell(0, 12, '', 0, 1)  # Spacing between sections

    filepath = os.path.join('data', filename)
    pdf.output(filepath, 'F')
    print(f"[✅] PDF saved to: {filepath}")

def main():
    pdf_filename = 'latest_news.pdf'
    pdf_filepath = os.path.join('data', pdf_filename)
    if os.path.exists(pdf_filepath):
        os.remove(pdf_filepath)

    urls = {
        "https://dbatu.ac.in/students-notice-board/": "Student Notice Board",
        "https://dbatu.ac.in/exam-section1/": "Exam Section",
        "https://dbatu.ac.in/registrar/": "Registrar",
        "https://dbatu.ac.in/events/": "Events",
        "https://dbatu.ac.in/fees-structure/": "Fees Structure",
        "https://dbatu.ac.in/student-corner/": "Student Corner",
        "https://dbatu.ac.in/admissions/": "Admissions",
        "https://dbatu.ac.in/academic-section-student-section-admission-contact-details/": "Academic & Admission Contacts",
        "https://dbatu.ac.in/academic-programs/": "Academic Programs",
        "https://dbatu.ac.in/research/": "Research",
        "https://cse.dbatu.ac.in/": "CSE Department"
    }

    all_news = {}

    for url, section_name in urls.items():
        print(f"[ℹ️] Scraping: {section_name}")
        latest_news = get_latest_news(url)
        all_news[section_name] = latest_news

    create_pdf(all_news, pdf_filename)
