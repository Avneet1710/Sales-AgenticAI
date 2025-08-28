# Munder Difflin Multi-Agent System

This repository contains the source code and documentation for a multi-agent system designed to automate sales, inventory, and quoting operations for the Munder Difflin Paper Company. The system uses the `smolagents` framework to orchestrate a team of specialized AI agents that handle customer inquiries and manage business logic.

## System Architecture

The system is built on a multi-agent architecture designed for a clear separation of duties, featuring one orchestrator agent and three specialized worker agents.

![Agent Workflow Diagram](AgenticAI%20architecture.png)

### Agent Roles

* **Orchestrator Agent**: This is the central "brain" of the operation. Implemented as a `CodeAgent`, its primary role is to receive and interpret customer requests. It then generates Python code at runtime to delegate tasks to the appropriate worker agent, manage the flow of information between them, and formulate the final response to the user.

* **Inventory Agent**: A `ToolCallingAgent` responsible for all inventory-related queries. Its duties include checking stock levels for specific items and providing reports on all available inventory.

* **Quoting Agent**: A `ToolCallingAgent` tasked with generating price quotes. It has access to tools that can search historical quote data to inform its pricing strategies.

* **Sales Agent**: A `ToolCallingAgent` that finalizes transactions. It is called to create a formal transaction in the database, which updates the company's financial records and inventory levels.

## File Structure

The repository is organized as follows:

```
.
├── AgenticAI architecture.png  # Visual diagram of the agent workflow
├── final_report.md             # The original project report
├── project_starter.py          # Main Python script with the agent implementation
├── quotes.csv                  # Historical quote data for the Quoting Agent
├── quote_requests.csv          # Sample of all incoming customer requests
├── quote_requests_sample.csv   # The specific test cases for evaluating the system
├── requirements.txt            # Required Python packages
├── terminal_log.txt            # A log of the test run execution
└── test_results.csv            # The final output from the evaluation run
```

## Setup and Usage

### 1. Install Dependencies

Ensure you have Python 3.8+ installed. You can install all required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
pip install 'smolagents[litellm]'
```

### 2. Set Up Environment Variables

Create a `.env` file in the root of the project and add your OpenAI-compatible API key:

```
UDACITY_OPENAI_API_KEY=your_openai_key_here
```

The system is configured to use the custom proxy at `https://openai.vocareum.com/v1`.

### 3. Run the System

To run the full test suite, execute the `project_starter.py` script from your terminal:

```bash
python project_starter.py
```

The script will initialize the database, process all requests from `quote_requests_sample.csv`, and generate a `test_results.csv` file with the outcome of each interaction.

## Performance and Evaluation

The system was evaluated against the criteria outlined in the project rubric, and it successfully met all requirements.

* **Cash Balance Changes**: The system correctly processed multiple sales, leading to at least three distinct changes in the `cash_balance` as recorded in `test_results.csv`.
* **Fulfilled Requests**: Multiple orders were successfully fulfilled, demonstrating the end-to-end functionality of the agent workflow.
* **Handled Exceptions**: The system correctly identified when items were out of stock and provided a clear reason for not fulfilling the request, as seen in the response for request #4.

### Strengths and Weaknesses

* **Strengths**: The final implementation demonstrates a robust and logical workflow. The `CodeAgent` orchestrator, guided by a highly detailed prompt, proved effective at interpreting user intent and delegating tasks correctly. The system's ability to check inventory before a sale is a key strength that prevents incorrect orders.
* **Weaknesses**: The development process highlighted the primary challenge of this architecture: the `CodeAgent`'s tendency to deviate from its instructions. This was only resolved through very specific, example-driven prompt engineering.

## Future Improvements

1. **Implement a More Robust Quoting Agent**: Enhance the `QuotingAgent` to perform dynamic pricing by analyzing inventory, historical data, and order size to generate quotes with strategic, calculated discounts.
2. **Develop a Pre-Processing Entity Extraction Layer**: Introduce a dedicated NLP model to parse user requests into a structured format (e.g., JSON) before they reach the orchestrator. This would dramatically simplify the orchestrator's task and reduce the potential for errors.
