import os
from typing import Annotated

import pandas as pd
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig

# Predefined order status for simulation
ORDER_STATUSES = {
    "12345": "Your order currently is being processed.",
    "67890": "Your order has been shipped and is on its way.",
    "54321": "Your order has been delivered.",
}


@tool
def check_order_status(order_id: str) -> str:
    """Check the status of an order given its ID."""
    return ORDER_STATUSES.get(order_id, "I couldn't find an order with that ID. Please check and try again.")


@tool
def request_human_assistance(name: str, email: str, phone: str, config: RunnableConfig) -> str:
    """Gather and save user contact information if they request a human representative or want to interact with a person."""
    out_filename = "user_contact_info.csv"
    data = {
        "id": [config.get("configurable", {}).get("thread_id")],
        "Full Name": [name],
        "Email": [email],
        "Phone Number": [phone],
    }
    df = pd.DataFrame(data)
    df.to_csv(out_filename, mode="a", index=False, header=not (os.path.isfile(out_filename)))
    return "Your information has been saved. A human representative will contact you soon."
