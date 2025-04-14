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

            Backend: Python, FastAPI
            Frontend: JavaScript, HTML/CSS
            Databases: MySQL, MongoDB
            Programming Languages: Python, C++
            Other Skills: Data Structures and Algorithms, Object-Oriented Programming, Git, Docker
            Your task is to write a compelling cold email to the recruiter regarding the job mentioned above.
            The email should:

            Directly align your technical skills and experience with the job requirements.
            Demonstrate how your expertise in Backend Development and Generative AI makes you a strong fit for the role.
            If applicable, subtly highlight any past projects or relevant work that strongly aligns with the job posting.
            Maintain a professional, concise, and engaging tone to maximize response chances.
            Avoid any preamble or self-introduction like “I am Apurva…” and instead dive directly into the value you bring to the company.
            Use strong action-oriented language to communicate how you can contribute effectively based on their job description.
            Your objective is to make the recruiter immediately see the direct value you offer for the position, increasing the probability of a positive response.
            Also add the most relevant ones from the following links to showcase Apurva's portfolio: {link_list}
            ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        print(chain_email)
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        print(res)
        return res.content
        

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))