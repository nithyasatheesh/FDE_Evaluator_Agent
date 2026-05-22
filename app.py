import streamlit as st
import pandas as pd
import zipfile
import io
import json
import hashlib
import os

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
# CACHE
# ==========================

CACHE_FILE="evaluation_cache.json"


def load_cache():

    try:

        if os.path.exists(
            CACHE_FILE
        ):

            with open(
                CACHE_FILE,
                "r"
            ) as f:

                return json.load(f)

    except:

        pass

    return {}


def save_cache(cache):

    try:

        with open(
            CACHE_FILE,
            "w"
        ) as f:

            json.dump(
                cache,
                f,
                indent=2
            )

    except:

        pass


CACHE=load_cache()


def build_cache_key(

    custom_prompt,

    rubric,

    submission

):

    combined=(

        custom_prompt+

        rubric+

        submission

    )

    return hashlib.sha256(

        combined.encode(
            errors="ignore"
        )

    ).hexdigest()


# ==========================
# READERS
# ==========================

def read_pdf(file):

    try:

        reader=PdfReader(file)

        pages=[]

        for p in reader.pages:

            txt=p.extract_text()

            if txt:

                pages.append(txt)

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

        output=[]

        for cell in nb.cells:

            if cell.cell_type=="markdown":

                output.append(
                    cell.source
                )

            elif cell.cell_type=="code":

                output.append(

                    "CODE:\n"+

                    cell.source

                )

        return "\n".join(output)

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

    rows=[]

    for _,r in df.iterrows():

        rows.append(

f"""
Criterion:
{r["Criterion"]}

Max Score:
{r["Max Score"]}

Description:
{r["Description"]}
"""

        )

    return "\n".join(rows)


# ==========================
# ZIP
# ==========================

def parse_submission(zip_bytes):

    result={

        "docs":[],

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
                    "docs"
                ].append(

                    read_pdf(
                        io.BytesIO(raw)
                    )

                )

            elif suffix==".docx":

                result[
                    "docs"
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
                    "docs"
                ].append(

                    read_html(
                        decoded
                    )

                )

            elif suffix==".py":

                result[
                    "code"
                ].append(
                    decoded
                )

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
                ].append(
                    decoded
                )

            elif suffix==".md":

                result[
                    "docs"
                ].append(
                    decoded
                )

        except:

            pass

    return result


# ==========================
# CONTEXT
# ==========================

def build_context(data):

    return f"""

DOCUMENTATION

{' '.join(data['docs'])[:12000]}

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

Maximum TOTAL=75.

Extract evidence FIRST.

Then assign rubric score.

Same evidence MUST produce same score.

No evidence=no score.

Never estimate.

Different quality MUST produce different scores.

Deduct:

- hardcoded logic
- duplicate code
- TODO comments
- boilerplate
- missing validation
- weak architecture
- weak modularity
- missing testing
- weak documentation
- missing security
- missing scalability

Code quality > project size.

Return ONLY JSON:

{

"evidence":{},

"scores":{},

"strengths":[],

"improvements":[]

}

User prompt overrides defaults.

"""

    response=client.chat.completions.create(

        model="gpt-4.1",

        temperature=0,

        top_p=0,

        response_format={

            "type":"json_object"

        },

        messages=[

        {

        "role":"system",

        "content":

        SYSTEM

        },

        {

        "role":"user",

        "content":

        prompt

        }

        ]

    )

    return response.choices[
        0
    ].message.content


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

"Strict Instructions"

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

    if problem.name.endswith(
        ".pdf"
    ):

        problem_text=read_pdf(
            problem
        )

    else:

        problem_text=read_docx(
            problem
        )

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

        cache_key=build_cache_key(

            custom_prompt,

            rubric_text,

            context

        )

        if cache_key in CACHE:

            result=CACHE[
                cache_key
            ]

        else:

            result=parse_json(

                evaluate_submission(
                    prompt
                )

            )

            CACHE[
                cache_key
            ]=result

            save_cache(
                CACHE
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

        factor=1

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

            row[
                f"{criterion} ({max_score})"
            ]=adjusted

            total+=adjusted

        if total>75:

            overflow=total-75

            cols=[

                c

                for c in row

                if "(" in c

            ]

            i=0

            while overflow>0:

                col=cols[
                    i%len(cols)
                ]

                if row[col]>0:

                    row[col]-=1

                    overflow-=1

                i+=1

            total=75

        row[
            "Total"
        ]=int(total)

        row[
            "Strengths"
        ]="; ".join(

            result.get(
                "strengths",
                []
            )

        )

        row[
            "Improvements"
        ]="; ".join(

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
