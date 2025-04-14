import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
                {job_description}

                ### INSTRUCTION:
                You are Apurva, a software engineer with 1+ year of experience specializing in Backend Development and Generative AI use cases.

                Your core technical skills include:
                - Backend: Python, FastAPI
                - Frontend: JavaScript, HTML/CSS
                - Databases: MySQL, MongoDB
                - Programming Languages: Python, C++
                - Other Skills: Data Structures and Algorithms, Object-Oriented Programming, Git, Docker

                Your task is to write a **concise and impactful cold email** to the recruiter for the job mentioned above.

                The email should:
                - Be **brief and skimmable**, ideally under 150 words.
                - **Immediately align your technical skills and project experience** with the job requirements.
                - **Avoid lengthy background or self-introduction** — get straight to the value you bring to the company.
                - Focus on how your **backend expertise and experience with Generative AI** make you a strong fit for the role.
                - Use **action-oriented and professional language** to convey confidence and clarity.
                - Include 1 **relevant and clearly labeled portfolio link** from this list: {link_list}
                - End with a **confident and polite call to action** (e.g., interest in discussing further).

                Avoid generic or verbose phrasing. Keep the tone crisp, professional, and easy to read for busy recruiters.
                EMAIL (NO PREAMBLE):
                """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content
        

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))