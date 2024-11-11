from concurrent.futures import ThreadPoolExecutor
from playwright.sync_api import sync_playwright
from deep_translator import GoogleTranslator
from datetime import datetime
from pandas import DataFrame
from lxml import html
import pandas as pd
import numpy as np
import random
import time
import re
import evpn
import os

proxies = [
    "185.188.76.152", "104.249.0.116", "185.207.96.76", "185.205.197.4", "185.199.117.103", "185.193.74.119", "185.188.79.150", "185.195.223.146", "181.177.78.203", "185.207.98.115", "186.179.10.253", "185.196.189.131", "185.205.199.143", "185.195.222.22", "186.179.20.88", "185.188.79.126", "185.195.213.198", "185.207.98.192", "186.179.27.166", "181.177.73.165", "181.177.64.160", "104.233.53.55", "185.205.197.152", "185.207.98.200", "67.227.124.192", "104.249.3.200", "104.239.114.248",
    "181.177.67.28", "185.193.74.7", "216.10.5.35", "104.233.55.126", "185.195.214.89", "216.10.1.63", "104.249.1.161", "186.179.27.91", "185.193.75.26", "185.195.220.100", "185.205.196.226", "185.195.221.9", "199.168.120.156", "181.177.69.174", "185.207.98.8", "185.195.212.240", "186.179.25.90", "199.168.121.162", "185.199.119.243", "181.177.73.168", "199.168.121.239", "185.195.214.176", "181.177.71.233", "104.233.55.230", "104.249.6.234", "104.249.3.87", "67.227.125.5", "104.249.2.53",
    "181.177.64.15", "104.249.7.79", "186.179.4.120", "67.227.120.39", "181.177.68.19", "186.179.12.120", "104.233.52.54", "104.239.117.252", "181.177.77.65", "185.195.223.56", "185.207.99.39", "104.249.7.103", "185.207.99.11", "186.179.3.220", "181.177.72.117", "185.205.196.180", "104.249.2.172", "185.207.98.181", "185.205.196.255", "104.239.113.239", "216.10.1.94", "181.177.77.2", "104.249.6.84", "104.239.115.50", "185.199.118.209", "104.233.55.92", "185.207.99.117", "104.233.54.71",
    "185.199.119.25", "181.177.78.82", "104.239.113.76", "216.10.7.90", "181.177.78.202", "104.239.119.189", "181.177.64.245", "185.199.118.216", "185.199.116.219", "185.188.77.64", "185.199.116.185", "185.188.78.176", "186.179.12.162", "185.205.197.193", "181.177.74.161", "67.227.126.121", "181.177.79.185",
]


# Function to remove Extra Spaces from Text
def remove_extra_spaces(_text: str):
    return ' '.join(_text.split())  # Remove extra spaces


# Function to extract 'alias' from 'name' string
def extract_alias(_text: str):
    alias = 'N/A'  # Default alias value
    if '(' in _text and ')' in _text:
        alias_text = _text.split('(')[1].split(')')[0]  # Splitting the 'name' value to extract alias
        if not alias_text.isnumeric():  # Checking if the extracted 'alias' is not a number
            return f"({alias_text})"  # Return alias_text
    return alias


def translate_text_with_retries(translator: GoogleTranslator, value: str, max_retries: int = 20):
    """Translate text with retry mechanism in case of errors."""
    retries = 0
    while retries < max_retries:
        try:
            # Attempt translation
            translated_value = translator.translate(text=value)
            return translated_value
        except Exception as e:
            retries += 1
            wait_time = random.uniform(a=2, b=5) * retries  # Exponential backoff with some randomness
            print(
                f"Error translating '{value}': {e}. Retrying in {wait_time:.2f} seconds (Attempt {retries}/{max_retries})")
            time.sleep(wait_time)
    return value  # Return original value if all retries fail


def translate_chunk_rows(chunk, columns: list):
    """Translate a chunk of rows in the dataframe for specified columns."""
    proxy = f"http://kunal_santani577-9elgt:QyqTV6XOSp@{random.choice(proxies)}:3199"
    print('Using Proxy:', proxy)
    translator = GoogleTranslator(source='iw', target='en', proxies={'http': proxy})

    for index, row in chunk.iterrows():
        for col in columns:
            if col in chunk.columns:
                value = row[col]
                try:
                    if isinstance(value, str) and value.strip().lower() == 'n/a':
                        chunk.at[index, col] = 'N/A'
                    elif isinstance(value, str) and value.strip() != '':
                        # Translate the value with retries
                        translated_value = translate_text_with_retries(translator, value)
                        chunk.at[index, col] = translated_value
                        print(f"Row {index}, Col '{col}': Translated '{value}' -> '{translated_value}'")
                except Exception as e:
                    print(f"Error translating '{value}' in row {index}, column '{col}': {e}")
                    chunk.at[index, col] = value  # Keep original value if error occurs
    return chunk


def translate_dataframe_in_chunks(df: DataFrame, columns: list, n_workers: int = 10):
    """Helper function to translate specified columns in the dataframe using parallel processing."""
    chunks = np.array_split(df, n_workers)
    print(f"Dataframe split into {n_workers} chunks for parallel processing.")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(lambda chunk: translate_chunk_rows(chunk, columns), chunks))

    print("All chunks processed.")
    return pd.concat(results)


def remove_specific_punctuation(_text):
    punctuation_marks = [
        ".", ",", "?", "!", ":", ";", "—", "-", "_", "(", ")", "[", "]", "{", "}", '"', "'", "‘", "’", "“", "”", "«",
        "»", "/", "\\", "|", "@", "#", "$", "%", "^", "&", "*", "+", "=", "~", "`", "<", ">", "…", "©", "®", "™"
    ]
    # Iterate over each punctuation mark and replace it in the original text
    if _text != 'N/A':
        for punc_mark in punctuation_marks:
            _text = _text.replace(punc_mark, '')
    return _text


# Function to clean text by removing unnecessary whitespace and line breaks
def text_cleaner(text: str) -> str:
    return re.sub(pattern=r"[\s\n\r]+", repl=" ", string=text).title().strip()


# Functions to extract specific data using XPath, with default values if data is missing
def get_title(profile_div, xpath_title):
    title = ' '.join(profile_div.xpath(xpath_title))
    # title = _title.replace('מ.', '')  # Removing 'from.' from string if it exists
    return title if title not in ['', 'אין מידע'] else 'N/A'


def get_procedure_no(profile_div, xpath_procedure_no):
    procedure_no = ' '.join(profile_div.xpath(xpath_procedure_no))
    return procedure_no if procedure_no not in ['', 'אין מידע'] else 'N/A'


def get_engaged_type(profile_div, xpath_engaged_type):
    engaged_type = text_cleaner(' '.join(profile_div.xpath(xpath_engaged_type)))
    return engaged_type if engaged_type not in ['', 'אין מידע'] else 'N/A'


def get_id_number(profile_div, xpath_id_number):
    _id_number = ' '.join(profile_div.xpath(xpath_id_number))
    return _id_number if _id_number not in ['', 'אין מידע'] else 'N/A'


def get_field_of_activity(profile_div, xpath_field_of_activity):
    field_of_activity = ' '.join(profile_div.xpath(xpath_field_of_activity))
    return field_of_activity if field_of_activity not in ['', 'אין מידע'] else 'N/A'


def get_nature_of_violation(profile_div, xpath_nature_of_violation):
    _nature_of_violation = ' '.join(profile_div.xpath(xpath_nature_of_violation))
    nature_of_violation = ' '.join(_nature_of_violation.split())  # To Remove extra Spaces
    return nature_of_violation if nature_of_violation not in ['', 'אין מידע'] else 'N/A'


def get_sanction_amount(profile_div, xpath_sanction_amount):
    sanction_amount = ' '.join(profile_div.xpath(xpath_sanction_amount))
    return sanction_amount if sanction_amount not in ['', 'אין מידע'] else 'N/A'


def get_reduction_clauses(profile_div, xpath_reduction_clauses):
    reduction_clauses = ' '.join(profile_div.xpath(xpath_reduction_clauses))
    return reduction_clauses if reduction_clauses not in ['', 'אין מידע'] else 'N/A'


def get_appeal(profile_div, xpath_appeal):
    appeal = text_cleaner(' '.join(profile_div.xpath(xpath_appeal)))
    return appeal if appeal not in ['', 'אין מידע'] else 'N/A'


def get_amount_of_penalty_after_appeal(profile_div, xpath_amount_of_penalty_after_appeal):
    amount_of_penalty_after_appeal = ' '.join(profile_div.xpath(xpath_amount_of_penalty_after_appeal))
    return amount_of_penalty_after_appeal if amount_of_penalty_after_appeal not in ['', 'אין מידע'] else 'N/A'


def get_pdf_url(profile_div, xpath_pdf_url):
    pdf_url = ' '.join(profile_div.xpath(xpath_pdf_url))
    return pdf_url if pdf_url not in ['', 'אין מידע'] else 'N/A'


class Scraper:
    def __init__(self):
        self.name = "gov_il"
        self.final_data = list()  # List to store extracted data from all profiles
        self.delivery_date = datetime.now().strftime('%Y%m%d')

        # Path to store the Excel file can be customized by the user
        self.excel_path = r"../Excel_Files"  # Client can customize their Excel file path here (default: govtsites > govtsites > Excel_Files)
        os.makedirs(self.excel_path, exist_ok=True)  # Create Folder if not exists
        self.filename = fr"{self.name}_{self.delivery_date}"  # Filename with Scrape Date

    # Function to scrape data from an individual HTML page
    def data_scraper(self, html_page):
        print('Scraping:', html_page.url)  # Print on which link scraper is working
        selector = html.fromstring(html_page.content())  # Parse the HTML content to apply Xpath and Scrape Data

        # Define XPaths to extract specific fields from the profile page
        xpath_title = './/div//h3[contains(@class, "ng-binding")]//text()'
        xpath_procedure_no = '''.//span[contains(@ng-bind-html, "item.Data.procedure_number")]//text()'''
        xpath_engaged_type = '''.//div[@id="MultiItemsemployee_type"]//text()'''
        xpath_id_number = '''.//span[@ng-bind-html="item.Data.id_number | dyDate:'d.M.yyyy':true"]//text()'''
        xpath_field_of_activity = '''.//span[@ng-bind-html="item.Data.biss_type | dyDate:'d.M.yyyy':true"]/text()'''
        xpath_nature_of_violation = '''.//span[@ng-bind-html="item.Data.hafara | dyDate:'d.M.yyyy':true"]/text()'''
        xpath_sanction_amount = '''.//span[@ng-bind-html="item.Data.itzum | dyDate:'d.M.yyyy':true"]/text()'''
        xpath_reduction_clauses = '''.//span[@ng-bind-html="item.Data.hafchata | dyDate:'d.M.yyyy':true"]//text()'''
        xpath_appeal = '''.//div[@id="MultiItemsirur"]//span[@ng-if]/text()'''
        xpath_amount_of_penalty_after_appeal = '''.//span[@ng-if="dynamicCtrl.Helpers.hasNoData(item.Data.sum__)" and normalize-space() != '']/text() | .//span[@ng-bind-html="item.Data.sum__ | dyDate:'d.M.yyyy':true" and normalize-space() != '']/text()'''
        xpath_pdf_url = '''.//div[@ng-repeat="file in item.Data.kovetz | limitTo:4"]/a/@href'''

        # Locate the list of profiles on current page using XPath
        xapth_profiles_list = '''//li[@ng-repeat="item in dynamicCtrl.ViewModel.dataResults"]'''
        profiles_list = selector.xpath(xapth_profiles_list)

        # Iterate through each profile and extract data using XPaths
        for profile_div in profiles_list:
            # Append the data to the final_data_native list
            name = get_title(profile_div, xpath_title)

            profile_dict = {
                'url': html_page.url,
                'name': name,
                'מספר הליך': get_procedure_no(profile_div, xpath_procedure_no),
                'סוג עוסק': get_engaged_type(profile_div, xpath_engaged_type),
                'ח"פ': get_id_number(profile_div, xpath_id_number),
                'תחום הפעילות': get_field_of_activity(profile_div, xpath_field_of_activity),
                'מהות ההפרה': get_nature_of_violation(profile_div, xpath_nature_of_violation),
                'סכום העיצום': get_sanction_amount(profile_div, xpath_sanction_amount),
                'סעיפי ההפחתה': get_reduction_clauses(profile_div, xpath_reduction_clauses),
                'ערעור': get_appeal(profile_div, xpath_appeal),
                'סכום העיצום לאחר ערעור': get_amount_of_penalty_after_appeal(profile_div, xpath_amount_of_penalty_after_appeal),
                'pdf_url': get_pdf_url(profile_div, xpath_pdf_url),
            }

            # Check if all keys except 'url' have 'N/A' value
            if not all(profile_dict[key] == 'N/A' for key in profile_dict if key != 'url'):
                self.final_data.append(profile_dict)  # Appending only if not all values are 'N/A'

        print('Data appended', '=' * 50)
        print(
            'waiting some time to avoid overloading the server...')  # Waiting to let the page load and find Next Page button is there is any
        time.sleep(1)  # Adding a delay to avoid overloading the server

    # Function to save the extracted data to an Excel file
    def save_to_excel(self):
        filename_native = f"{self.excel_path}/{self.filename}_native.xlsx"
        print("Converting List of Dictionaries into DataFrame then into Excel file...")
        try:
            print("Generating Native Excel file...")
            native_df: DataFrame = pd.DataFrame(self.final_data)  # Convert data into a pandas DataFrame
            native_df.drop_duplicates(inplace=True)  # Removing Duplicate data from DataFrame

            # native_df['id'] = range(1, len(native_df) + 1)  # Add an ID column
            # native_df.set_index(keys='id', inplace=True)  # Set the 'id' column as index

            filename_english = f"{self.excel_path}/{self.filename}_english.xlsx"
            try:
                print("Creating English sheet...")

                columns_to_translate = ["name", "סוג עוסק", 'ח"פ', "תחום הפעילות", "מהות ההפרה", "סכום העיצום",
                                        "סעיפי ההפחתה", "ערעור", "סכום העיצום לאחר ערעור"]
                # Perform chunked translation with parallel processing
                translated_df = translate_dataframe_in_chunks(native_df, columns_to_translate)

                # Create new 'alias' column based on 'name' column
                native_df['alias'] = native_df['name'].apply(extract_alias)
                translated_df['alias'] = translated_df['name'].apply(extract_alias)

                # columns_to_translate = ["alias"]
                # # Perform chunked translation with parallel processing
                # translated_df['alias'] = translate_dataframe_in_chunks(native_df['alias'], columns_to_translate)

                # Perform str.replace to remove alias value from the name column for each row
                # native_df['name'] = native_df.apply(lambda row: row['name'].replace(row['alias'], ''), axis=1)
                native_df['name'] = native_df.apply(lambda row: row['name'].replace(row['alias'], '') if row['name'] != 'N/A' else row['name'], axis=1)

                # Remove 'alias' column first if it already exists, to avoid conflicts
                alias_col_native = native_df.pop('alias')
                # Insert 'alias' column at the second position (index 1)
                native_df.insert(loc=2, column='alias', value=alias_col_native)

                native_df['name'] = native_df['name'].apply(remove_specific_punctuation)  # Remove Punctuations from 'name' column for every row
                native_df['alias'] = native_df['alias'].apply(remove_specific_punctuation)  # Remove Punctuations from 'alias' column for every row

                native_df['name'] = native_df['name'].apply(remove_extra_spaces)  # Remove Extra spaces from 'name'
                native_df['alias'] = native_df['alias'].apply(remove_extra_spaces)  # Remove Extra spaces from 'alias'

                with pd.ExcelWriter(path=filename_native, engine='xlsxwriter') as writer:
                    native_df.to_excel(excel_writer=writer, index=False)  # Write data to Native Excel file
                print("Native Excel file Successfully created.")

                # Translate column headers
                translated_column_mapping = {
                    'מספר הליך': 'procedure_number',
                    'סוג עוסק': 'engaged_type',
                    'ח"פ': 'id_number',
                    'תחום הפעילות': 'field_of_activity',
                    'מהות ההפרה': 'nature_of_violation',
                    'סכום העיצום': 'sanction_amount',
                    'סעיפי ההפחתה': 'reduction_clauses',
                    'ערעור': 'appeal',
                    'סכום העיצום לאחר ערעור': 'amount_of_penalty_after_appeal',
                }

                '''-------------------------CLEANING SECTION-------------------------'''
                # Rename columns
                translated_df.rename(columns=translated_column_mapping, inplace=True)
                # translated_df['id'] = range(1, len(translated_df) + 1)  # Add an ID column
                # translated_df.set_index(keys='id', inplace=True)  # Set the 'id' column as index

                # Remove 'alias' column first if it already exists, to avoid conflicts
                alias_col_translated = translated_df.pop('alias')
                # Insert 'alias' column at the second position (index 1)
                translated_df.insert(loc=2, column='alias', value=alias_col_translated)

                # Perform str.replace to remove alias value from the name column for each row
                translated_df['name'] = translated_df.apply(lambda row: row['name'].replace(row['alias'], '') if row['name'] != 'N/A' else row['name'], axis=1)

                translated_df['name'] = translated_df['name'].apply(
                    remove_specific_punctuation)  # Remove Punctuations from 'name' column for every row
                translated_df['alias'] = translated_df['alias'].apply(
                    remove_specific_punctuation)  # Remove Punctuations from 'alias' column for every row

                translated_df['name'] = translated_df['name'].apply(remove_extra_spaces)  # Remove Extra spaces from 'name'
                translated_df['alias'] = translated_df['alias'].apply(remove_extra_spaces)  # Remove Extra spaces from 'alias'
                with pd.ExcelWriter(path=filename_english, engine='xlsxwriter') as writer:
                    translated_df.to_excel(excel_writer=writer, index=False)  # Write data to English Excel file
                    print("English Excel file Successfully created.")
            except Exception as e:
                print('Error while Generating English Excel file:', e)
        except Exception as e:
            print('Error while Generating Native Excel file:', e)

    # Main function to start the scraping process
    def start_scraping(self):
        print('Scraper started...')
        # logging.log(msg='Scraper started...', level=0)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, timeout=100000)  # Launch Chromium browser
            context = browser.new_context(ignore_https_errors=True)  # Create a new browser context
            page = context.new_page()  # Open a new page
            link = 'https://www.gov.il/he/Departments/DynamicCollectors/monetary_sunction_consumer_protection?skip=0'
            page.goto(link, timeout=100000)  # Navigate to the target URL
            self.data_scraper(html_page=page)  # Scrape data from the first page

            # Loop to handle pagination and scrape subsequent pages
            while True:
                try:
                    # Look for the "Next" button to proceed to the next page
                    next_page_button = page.locator(
                        '//div[@class="col-5 col-lg-2 btn-next"]/a[not (contains(@class, "ng-hide")) and @aria-hidden = "false"]')
                    if next_page_button.is_visible():
                        print(f"Next page button found, sending Request on next Page...")
                        next_page_button.click()
                        time.sleep(2)  # Wait for the next page to load
                        self.data_scraper(html_page=page)  # Scrape data from the new page
                    else:
                        print("Next page button not found, ending pagination!")
                        break  # Exit loop if "Next" button is Not found
                except Exception as e:
                    print('Error in pagination:', e)
                    break

            self.save_to_excel()  # Save the scraped data to Excel file


if __name__ == '__main__':
    # Record the start time
    start_time = time.time()
    api = evpn.ExpressVpnApi()
    api.connect('105')
    time.sleep(5)
    # Your program or function code here
    scraper = Scraper()
    scraper.start_scraping()  # Start the scraping process

    # Record the end time
    end_time = time.time()
    # Calculate the total execution time
    execution_time = end_time - start_time
    api.disconnect()
    print(f"Total execution time: {execution_time} seconds")
