import streamlit as st
import pandas as pd
import zipfile
import io
import json
import re

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from bs4 import BeautifulSoup
from docx import Document
from PyPDF2 import PdfReader
import nbformat


# ==========================
# CONFIG
# ==========================

st.set_page_config(
    page_title="AI Case Study Evaluator",
    layout="wide"
)

st.title("📊 AI Case Study Evaluator")

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)


# ==========================
# FILE READERS
# ==========================

def read_pdf(file):

    try:

        reader = PdfReader(file)

        pages=[]

        for page in reader.pages:

            t=page.extract_text()

            if t:

                pages.append(t)

        return "\n".join(pages)

    except:

        return ""


def read_docx(file):

    try:

        doc=Document(file)

        return "\n".join(

            p.text

            for p in doc.paragraphs

        )

    except:

        return ""


def read_html(text):

    try:

        soup=BeautifulSoup(
            text,
            "html.parser"
        )

        for tag in soup(
            ["script","style"]
        ):

            tag.decompose()

        return soup.get_text(
            separator="\n"
        )

    except:

        return ""


def read_notebook(text):

    try:

        nb=nbformat.reads(
            text,
            as_version=4
        )

        out=[]

        for cell in nb.cells:

            if cell.cell_type=="markdown":

                out.append(
                    cell.source
                )

            elif cell.cell_type=="code":

                out.append(

                    "CODE:\n"+

                    cell.source

                )

        return "\n".join(out)

    except:

        return ""


def summarize_csv(text):

    try:

        df=pd.read_csv(
            io.StringIO(text)
        )

        return f"""

Rows:
{df.shape[0]}

Columns:
{list(df.columns)}

Sample:

{df.head(3).to_string()}

"""

    except:

        return ""


# ==========================
# RUBRIC
# ==========================

def rubric_to_text(df):

    text=[]

    for _,r in df.iterrows():

        text.append(

f"""
Criterion:
{r["Criterion"]}

Max Score:
{r["Max Score"]}

Description:
{r["Description"]}
"""

        )

    return "\n".join(text)


# ==========================
# ZIP PARSER
# ==========================

def parse_submission(zip_bytes):

    result={

        "documentation":[],

        "code":[],

        "notebooks":[],

        "database":[],

        "datasets":[]

    }

    z=zipfile.ZipFile(
        io.BytesIO(zip_bytes)
    )

    for file in z.namelist():

        try:

            raw=z.read(file)

            suffix=Path(
                file
            ).suffix.lower()

            decoded=raw.decode(
                errors="ignore"
            )

            if suffix==".pdf":

                result[
                    "documentation"
                ].append(

                    read_pdf(
                        io.BytesIO(raw)
                    )

                )

            elif suffix==".docx":

                result[
                    "documentation"
                ].append(

                    read_docx(
                        io.BytesIO(raw)
                    )

                )

            elif suffix in [

                ".html",
                ".htm"

            ]:

                result[
                    "documentation"
                ].append(

                    read_html(
                        decoded
                    )

                )

            elif suffix==".py":

                result[
                    "code"
                ].append(decoded)

            elif suffix==".ipynb":

                result[
                    "notebooks"
                ].append(

                    read_notebook(
                        decoded
                    )

                )

            elif suffix==".csv":

                result[
                    "datasets"
                ].append(

                    summarize_csv(
                        decoded
                    )

                )

            elif suffix==".sql":

                result[
                    "database"
                ].append(decoded)

            elif suffix==".md":

                result[
                    "documentation"
                ].append(decoded)

        except:

            pass

    return result


# ==========================
# CONTEXT
# ==========================

def build_context(data):

    return f"""

DOCUMENTATION

{' '.join(data['documentation'])[:12000]}

NOTEBOOKS

{' '.join(data['notebooks'])[:10000]}

CODE

{' '.join(data['code'])[:18000]}

DATABASE

{' '.join(data['database'])[:5000]}

DATASETS

{data['datasets']}

"""


# ==========================
# OPENAI
# ==========================

def evaluate_submission(prompt):

    SYSTEM="""

STRICT evaluator.

For identical submissions produce identical scores.

Never randomly vary rubric scores.

Highest TOTAL score=75.

Score ONLY evidence.

No evidence=no score.

Extract evidence FIRST.

Then score.

Return ONLY JSON:

{
"evidence":{},
"scores":{},
"strengths":[],
"improvements":[]
}

Weak implementation:

0-49

Average:

50-59

Good:

60-65

Excellent:

65-69

Exceptional:

70-75 ONLY

Deduct:

- boilerplate
- TODO
- hardcoded
- weak architecture
- duplicate code
- missing testing
- missing validation
- missing docs
- weak modularity

User prompt overrides defaults.

"""

    response=client.chat.completions.create(

        model="gpt-4.1",

        temperature=0,

        response_format={
            "type":"json_object"
        },

        messages=[

        {

        "role":"system",

        "content":SYSTEM

        },

        {

        "role":"user",

        "content":prompt

        }

        ]

    )

    return response.choices[
        0
    ].message.content


# ==========================
# JSON
# ==========================

def parse_json(raw):

    try:

        return json.loads(raw)

    except:

        return {

            "scores":{},
            "strengths":[],
            "improvements":[]

        }


# ==========================
# UI
# ==========================

problem=st.file_uploader(
"Problem",
["pdf","docx"]
)

rubric=st.file_uploader(
"Rubric",
["xlsx"]
)

submissions=st.file_uploader(
"Participant ZIP",
type=["zip"],
accept_multiple_files=True
)

custom_prompt=st.text_area(
"Strict Evaluation Rules"
)


# ==========================
# RUN
# ==========================

if st.button("Evaluate"):

    rubric_df=pd.read_excel(
        rubric
    )

    rubric_text=rubric_to_text(
        rubric_df
    )

    problem_text=read_pdf(
        problem
    ) if problem.name.endswith(
        ".pdf"
    ) else read_docx(problem)

    def process(zip_file):

        parsed=parse_submission(
            zip_file.read()
        )

        context=build_context(
            parsed
        )

        prompt=f"""

{custom_prompt}

PROBLEM

{problem_text}

RUBRIC

{rubric_text}

SUBMISSION

{context}

"""

        result=parse_json(

            evaluate_submission(
                prompt
            )

        )

        row={

            "Participant":
            zip_file.name

        }

        raw=[]

        raw_total=0

        for _,r in rubric_df.iterrows():

            criterion=r[
                "Criterion"
            ]

            max_score=int(
                r["Max Score"]
            )

            score=float(

                result.get(
                    "scores",
                    {}
                ).get(
                    criterion,
                    0
                )

            )

            score=max(

                0,

                min(
                    score,
                    max_score
                )

            )

            raw.append(

                (
                    criterion,
                    max_score,
                    score
                )

            )

            raw_total+=score

        factor=1.0

        if raw_total>75:

            factor=75/raw_total

        factor=float(
            f"{factor:.4f}"
        )

        total=0

        for criterion,max_score,score in raw:

            adjusted=int(

                score*factor

            )

            adjusted=max(

                0,

                min(
                    adjusted,
                    max_score
                )

            )

            row[
                f"{criterion} ({max_score})"
            ]=adjusted

            total+=adjusted

        if total>75:

            total=75

        row["Total"]=int(total)

        row["Strengths"]="; ".join(

            result.get(
                "strengths",
                []
            )

        )

        row["Improvements"]="; ".join(

            result.get(
                "improvements",
                []
            )

        )

        return row

    with ThreadPoolExecutor(
        max_workers=4
    ) as executor:

        results=list(

            executor.map(
                process,
                submissions
            )

        )

    df=pd.DataFrame(
        results
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    excel=io.BytesIO()

    with pd.ExcelWriter(

        excel,

        engine="xlsxwriter"

    ) as writer:

        df.to_excel(
            writer,
            index=False
        )

    st.download_button(

        "Download Excel",

        excel.getvalue(),

        "evaluation_report.xlsx"

    )
