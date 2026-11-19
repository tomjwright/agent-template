"""The job application finder: finds roles, tailors a resume, drafts and
submits an application.

    JobFinderAgent -> ResumeImproverAgent -> JobApplicationAgent

The supervisor routes between them; each agent has only the tools it needs.
Replace the canned tools in tools.py with real APIs for a real project.
"""

from langchain_core.language_models import BaseChatModel

from agent_template.agents import Agent
from agent_template.llm import chat_model
from agent_template.orchestrator import Orchestrator
from examples.job_finder.tools import improve_resume_section, search_jobs, submit_application

APP_NAME = "job_finder"


def build_orchestrator(model: BaseChatModel | None = None) -> Orchestrator:
    model = model or chat_model()
    agents = [
        Agent(
            name="JobFinderAgent",
            description="Searches job boards for roles matching the user's criteria.",
            system_prompt=(
                "You find jobs. Use the search_jobs tool with the user's criteria. "
                "Report the best matching jobs with title, company, location and url."
            ),
            tools=[search_jobs],
        ),
        Agent(
            name="ResumeImproverAgent",
            description="Tailors the user's resume to a specific job listing.",
            system_prompt=(
                "You improve resumes. Given the chosen job from the conversation, call "
                "improve_resume_section with a resume section and the job's skills. "
                "Return the tailored resume section."
            ),
            tools=[improve_resume_section],
        ),
        Agent(
            name="JobApplicationAgent",
            description="Writes a cover letter and submits the application for a chosen job.",
            system_prompt=(
                "You draft applications. Write a short cover letter for the job discussed, "
                "then call submit_application with the job url, resume and letter. "
                "Finish with a summary for the user."
            ),
            tools=[submit_application],
        ),
    ]
    return Orchestrator(agents, model=model)
