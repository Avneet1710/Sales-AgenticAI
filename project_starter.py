import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine
from smolagents import CodeAgent, ToolCallingAgent, Tool, LiteLLMModel

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper", "category": "paper", "unit_price": 0.05},
    {"item_name": "Letter-sized paper", "category": "paper", "unit_price": 0.06},
    {"item_name": "Cardstock", "category": "paper", "unit_price": 0.15},
    {"item_name": "Colored paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Glossy paper", "category": "paper", "unit_price": 0.20},
    {"item_name": "Matte paper", "category": "paper", "unit_price": 0.18},
    {"item_name": "Recycled paper", "category": "paper", "unit_price": 0.08},
    {"item_name": "Eco-friendly paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Poster paper", "category": "paper", "unit_price": 0.25},
    {"item_name": "Banner paper", "category": "paper", "unit_price": 0.30},
    {"item_name": "Kraft paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Construction paper", "category": "paper", "unit_price": 0.07},
    {"item_name": "Wrapping paper", "category": "paper", "unit_price": 0.15},
    {"item_name": "Glitter paper", "category": "paper", "unit_price": 0.22},
    {"item_name": "Decorative paper", "category": "paper", "unit_price": 0.18},
    {"item_name": "Letterhead paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Legal-size paper", "category": "paper", "unit_price": 0.08},
    {"item_name": "Crepe paper", "category": "paper", "unit_price": 0.05},
    {"item_name": "Photo paper", "category": "paper", "unit_price": 0.25},
    {"item_name": "Uncoated paper", "category": "paper", "unit_price": 0.06},
    {"item_name": "Butcher paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Heavyweight paper", "category": "paper", "unit_price": 0.20},
    {"item_name": "Standard copy paper", "category": "paper", "unit_price": 0.04},
    {"item_name": "Bright-colored paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Patterned paper", "category": "paper", "unit_price": 0.15},
    # Product Types (priced per unit)
    {"item_name": "Paper plates", "category": "product", "unit_price": 0.10}, # per plate
    {"item_name": "Paper cups", "category": "product", "unit_price": 0.08}, # per cup
    {"item_name": "Paper napkins", "category": "product", "unit_price": 0.02}, # per napkin
    {"item_name": "Disposable cups", "category": "product", "unit_price": 0.10}, # per cup
    {"item_name": "Table covers", "category": "product", "unit_price": 1.50}, # per cover
    {"item_name": "Envelopes", "category": "product", "unit_price": 0.05}, # per envelope
    {"item_name": "Sticky notes", "category": "product", "unit_price": 0.03}, # per sheet
    {"item_name": "Notepads", "category": "product", "unit_price": 2.00}, # per pad
    {"item_name": "Invitation cards", "category": "product", "unit_price": 0.50}, # per card
    {"item_name": "Flyers", "category": "product", "unit_price": 0.15}, # per flyer
    {"item_name": "Party streamers", "category": "product", "unit_price": 0.05}, # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20}, # per roll
    {"item_name": "Paper party bags", "category": "product", "unit_price": 0.25}, # per bag
    {"item_name": "Name tags with lanyards", "category": "product", "unit_price": 0.75}, # per tag
    {"item_name": "Presentation folders", "category": "product", "unit_price": 0.50}, # per folder
    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},
    # Specialty papers
    {"item_name": "100 lb cover stock", "category": "specialty", "unit_price": 0.50},
    {"item_name": "80 lb text paper", "category": "specialty", "unit_price": 0.40},
    {"item_name": "250 gsm cardstock", "category": "specialty", "unit_price": 0.30},
    {"item_name": "220 gsm poster paper", "category": "specialty", "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.
    """
    np.random.seed(seed)
    num_items = int(len(paper_supplies) * coverage)
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )
    selected_items = [paper_supplies[i] for i in selected_indices]
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),
            "min_stock_level": np.random.randint(50, 150)
        })
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:
    """
    Set up the Munder Difflin database with all required tables and initial records.
    """
    try:
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],
            "units": [],
            "price": [],
            "transaction_date": [],
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        initial_date = datetime(2025, 1, 1).isoformat()

        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        initial_transactions = []
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales'.
    """
    try:
        date_str = date.isoformat() if isinstance(date, datetime) else date
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.
    """
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.
    """
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.
    """
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    delivery_date_dt = input_date_dt + timedelta(days=days)
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.
    """
    try:
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0

def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.
    """
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    cash = get_cash_balance(as_of_date)
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }

def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.
    """
    conditions = []
    params = {}

    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row) for row in result.mappings()]

# dotenv.load_dotenv()
API_KEY = "voc-10166986601587664276185689c6aaf2cfae0.86305996"

llm_model = LiteLLMModel(
    model_id="gpt-3.5-turbo",
    api_base="https://openai.vocareum.com/v1",
    api_key=API_KEY,
    temperature=0.1,
)

# 2. Define Tools for the Agents
class InventoryTools(Tool):
    """Container for all inventory-related tools."""

    @staticmethod
    def get_all_inventory(as_of_date: str) -> str:
        """
        Retrieves a snapshot of all available inventory as of a specific date.
        Args:
            as_of_date (str): The date for the inventory snapshot in YYYY-MM-DD format.
        Returns:
            str: A string representation of the inventory dictionary (item_name: stock_level).
        """
        inventory_dict = get_all_inventory(as_of_date)
        return str(inventory_dict)

    @staticmethod
    def get_stock_level(item_name: str, as_of_date: str) -> int:
        """
        Checks the stock level for a single, specific item on a given date.
        Args:
            item_name (str): The name of the item to check.
            as_of_date (str): The date of the check in YYYY-MM-DD format.
        Returns:
            int: The current stock level of the item.
        """
        stock_df = get_stock_level(item_name, as_of_date)
        if not stock_df.empty:
            return int(stock_df.iloc[0]["current_stock"])
        return 0

    @staticmethod
    def get_supplier_delivery_date(item_name: str, quantity: int, as_of_date: str) -> str:
        """
        Estimates the delivery date for a new stock order of a specific item.
        Args:
            item_name (str): The name of the item being ordered.
            quantity (int): The quantity of the item being ordered.
            as_of_date (str): The date the order is placed, in YYYY-MM-DD format.
        Returns:
            str: The estimated delivery date in YYYY-MM-DD format.
        """
        return get_supplier_delivery_date(as_of_date, quantity)

class QuotingTools(Tool):
    """Container for all quoting-related tools."""

    @staticmethod
    def search_quote_history(search_terms: List[str], limit: int = 5) -> str:
        """
        Searches historical quotes for similar requests to inform new pricing.
        Args:
            search_terms (List[str]): A list of keywords from the customer's request.
            limit (int): The maximum number of historical quotes to return.
        Returns:
            str: A string representation of the list of matching historical quotes.
        """
        history = search_quote_history(search_terms, limit)
        return str(history)

class SalesTools(Tool):
    """Container for all sales-related tools."""

    @staticmethod
    def create_transaction(item_name: str, transaction_type: str, quantity: int, price: float, date: str) -> str:
        """
        Finalizes a sale by creating a transaction record in the database.
        Args:
            item_name (str): The name of the item sold.
            transaction_type (str): The type of transaction, should be 'sales'.
            quantity (int): The number of units sold.
            price (float): The total price of the sale.
            date (str): The date of the transaction in YYYY-MM-DD format.
        Returns:
            str: A confirmation message with the new transaction ID.
        """
        transaction_id = create_transaction(item_name, transaction_type, quantity, price, date)
        return f"Successfully created transaction with ID: {transaction_id}"

    @staticmethod
    def get_cash_balance(as_of_date: str) -> float:
        """
        Retrieves the current cash balance of the company.
        Args:
            as_of_date (str): The date for which to calculate the balance.
        Returns:
            float: The current cash balance.
        """
        return get_cash_balance(as_of_date)

# 3. Create the Worker Agents
inventory_agent = ToolCallingAgent(
    name="InventoryAgent",
    description="You are an inventory management specialist. Your role is to check stock levels, report on all available inventory, and estimate supplier delivery dates. You are precise and provide only the data requested.",
    tools=[InventoryTools()],
    model=llm_model,
)

quoting_agent = ToolCallingAgent(
    name="QuotingAgent",
    description="You are a quoting specialist. Your job is to create competitive and attractive price quotes for customers. You should consult historical quote data to inform your pricing and always check for product availability before providing a quote. You should apply bulk discounts strategically to encourage sales.",
    tools=[QuotingTools()],
    model=llm_model,
)

sales_agent = ToolCallingAgent(
    name="SalesAgent",
    description="You are a sales finalization specialist. Your responsibility is to process a confirmed sale by creating a transaction in the database. You must be accurate with the details of the transaction.",
    tools=[SalesTools()],
    model=llm_model,
)

# 4. Create the Orchestration Agent
orchestrator_agent = CodeAgent(
    name="OrchestratorAgent",
    description=(
        "You are the orchestrator for the Beaver's Choice Paper Company. "
        "Your primary role is to manage a team of specialist agents to handle customer requests efficiently. "
        "Here is your workflow: "
        "1. Receive the customer request. "
        "2. If the request is a simple inventory check, delegate it to the 'InventoryAgent'. "
        "3. If the request requires a price quote, delegate it to the 'QuotingAgent'. The QuotingAgent will need to know what is in stock, so you must first use the 'InventoryAgent' to get the current inventory and pass that information along with the original request to the 'QuotingAgent'. "
        "4. If the customer's request implies they have accepted a quote and want to proceed with a purchase, delegate the task to the 'SalesAgent' to finalize the transaction. "
        "5. You are the only agent that communicates with the user. You will synthesize the information from your team and provide the final, clear response."
    ),
    managed_agents=[inventory_agent, quoting_agent, sales_agent],
    model=llm_model,
)

def call_your_multi_agent_system(request_with_date: str):
    """
    This function takes the user's request and passes it to the orchestrator agent.
    """
    print(f"--- Passing request to Orchestrator Agent ---")
    response = orchestrator_agent.run(request_with_date)
    return response


# Run your test scenarios by writing them here. Make sure to keep track of them.
def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        response = call_your_multi_agent_system(request_with_date)

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
