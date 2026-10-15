"""Employee Attendance & Analytics API — Parts A, B, and C.

Run: uvicorn app.main:app --port 8000
Configuration: MONGO_URI and MONGO_DB; environment variables override .env.
"""
import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from bson import json_util
from bson.decimal128 import Decimal128
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from pydantic import AfterValidator, BaseModel, Field, StringConstraints, model_validator
from pymongo import AsyncMongoClient, IndexModel, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

load_dotenv()

UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Each worker creates its client on the event loop that will use it.
    client = AsyncMongoClient(
        os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        tz_aware=True,
        maxPoolSize=100,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        waitQueueTimeoutMS=10000,
    )
    app.state.db = client[os.getenv("MONGO_DB", "attendance_db")]
    try:
        await create_indexes(app.state.db)
        yield
    finally:
        await client.close()


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


async def create_indexes(db):
    await db.employees.create_indexes([
        IndexModel([("emp_code", 1)], unique=True, name="employee_code_unique"),
        IndexModel([("department", 1), ("emp_code", 1)], name="department_employee"),
        IndexModel([("joined_on", 1)], name="employee_joined_on"),
    ])
    await db.attendance_logs.create_indexes([
        IndexModel([("emp_code", 1), ("date", -1)], unique=True, name="employee_date_unique"),
        IndexModel([("date", -1), ("emp_code", 1)], name="date_employee"),
        IndexModel([("status", 1), ("date", -1), ("emp_code", 1)], name="status_date_employee"),
    ])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def utc_seconds(value: datetime) -> datetime:
    # A naive BSON datetime represents UTC, never the machine's local timezone.
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).replace(microsecond=0)


def epoch_millis(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return (value.astimezone(UTC) - EPOCH) // timedelta(milliseconds=1)


def now_millis() -> int:
    return epoch_millis(utc_seconds(datetime.now(UTC)))


def from_epoch_millis(value: int) -> datetime:
    return EPOCH + timedelta(seconds=value // 1000)


def attendance_date(punch_in: datetime, shift_start: str, shift_end: str) -> str:
    local = utc_seconds(punch_in).astimezone(IST)
    day = local.date()
    if shift_end <= shift_start and local.time() < time.fromisoformat(shift_end):
        day -= timedelta(days=1)
    return day.isoformat()


def shift_boundaries(date_str: str, shift_start: str, shift_end: str):
    day = date.fromisoformat(date_str)
    start = datetime.combine(day, time.fromisoformat(shift_start), tzinfo=IST)
    end = datetime.combine(day, time.fromisoformat(shift_end), tzinfo=IST)
    if shift_end <= shift_start:
        end += timedelta(days=1)
    return start, end


def compute_late_minutes(punch_in: datetime, shift_start: str, date_str: str) -> int:
    start = datetime.combine(date.fromisoformat(date_str), time.fromisoformat(shift_start), tzinfo=IST)
    seconds = (utc_seconds(punch_in) - start) // timedelta(seconds=1)
    return seconds // 60 if seconds > 600 else 0


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    seconds = (utc_seconds(punch_out) - utc_seconds(punch_in)) // timedelta(seconds=1)
    return float((Decimal(seconds) / Decimal(3600)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def compute_overtime(punch_out: datetime, shift_start: str, shift_end: str, date_str: str) -> int:
    _, end = shift_boundaries(date_str, shift_start, shift_end)
    minutes = (utc_seconds(punch_out) - end) // timedelta(minutes=1)
    return minutes if minutes >= 30 else 0


def serialize(value):
    """Convert instants, including nested audit changes, at the HTTP boundary."""
    if isinstance(value, datetime):
        return epoch_millis(value)
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items() if key != "_id"}
    if isinstance(value, list):
        return [serialize(item) for item in value]
    return value


def attendance_response(doc):
    record = {key: doc[key] for key in ("emp_code", "date", "status")}
    for key in ("punch_in", "punch_out", "work_hours"):
        record[key] = doc.get(key)
    for key in ("late_minutes", "overtime_minutes"):
        record[key] = doc.get(key, 0)
    record["half_day"] = doc.get("half_day", False)
    record["history"] = doc.get("history", [])
    return serialize(record)


def valid_calendar_date(value: str) -> str:
    date.fromisoformat(value)
    return value


def valid_month(value: str) -> str:
    # The YYYY-MM pattern is checked by StringConstraints before this runs.
    # Reject year 0000 as a request validation error; Python's date type does
    # not support it and date construction would otherwise fail later.
    if value.startswith("0000-"):
        raise ValueError("year 0000 is not a valid calendar year")
    return value


def attendance_query(emp_code, date_from, date_to, status):
    if date_from is not None and date_to is not None and date_from > date_to:
        raise RequestValidationError([{
            "type": "value_error", "loc": ("query", "date_to"),
            "msg": "date_to must be on or after date_from", "input": date_to,
        }])
    query = {}
    if emp_code is not None:
        query["emp_code"] = emp_code
    if date_from is not None or date_to is not None:
        query["date"] = {}
        if date_from is not None:
            query["date"]["$gte"] = date_from
        if date_to is not None:
            query["date"]["$lte"] = date_to
    if status is not None:
        query["status"] = status
    index = "employee_date_unique" if emp_code is not None else (
        "status_date_employee" if status is not None else "date_employee"
    )
    return query, index


def attendance_list_operation(emp_code, date_from, date_to, status, page, page_size):
    query, index = attendance_query(emp_code, date_from, date_to, status)
    return {
        "filter": query,
        "sort": {"date": -1, "emp_code": 1},
        "skip": (page - 1) * page_size,
        "limit": page_size,
        "hint": index,
    }


def month_bounds(month: str):
    if int(month[:4]) == 0:
        raise ValueError("year 0000 is not a valid calendar year")
    start = date.fromisoformat(f"{month}-01")
    if month == "9999-12":
        # Calendar dates are stored as fixed-width YYYY-MM-DD strings. This
        # sentinel sorts after every valid date in December 9999 and avoids
        # constructing the unrepresentable date 10000-01-01.
        return start.isoformat(), "9999-12-32"
    end = date(start.year + (start.month == 12), start.month % 12 + 1, 1)
    return start.isoformat(), end.isoformat()


def round_half_up_expression(value, places: int):
    scale = 10 ** places
    return {"$toDouble": {"$divide": [
        {"$floor": {"$add": [
            {"$multiply": [{"$toDecimal": value}, Decimal128(str(scale))]},
            Decimal128("0.5"),
        ]}},
        Decimal128(str(scale)),
    ]}}


def working_days_expression(joined_on, start: str, end: str):
    """Count weekdays from max(joined_on, period start) up to period end."""
    final_month = end == "9999-12-32"
    final_date = "9999-12-31" if final_month else end
    beyond_period = {"$gt": [joined_on, final_date]} if final_month else {"$gte": [joined_on, end]}
    effective_start = {"$cond": [{"$gt": [joined_on, start]}, joined_on, start]}
    start_date = {"$dateFromString": {"dateString": "$$effective_start", "format": "%Y-%m-%d"}}
    end_date = {"$dateFromString": {"dateString": final_date, "format": "%Y-%m-%d"}}
    days_between = {"$dateDiff": {"startDate": start_date, "endDate": end_date, "unit": "day"}}
    day_count = {"$add": [days_between, 1]} if final_month else days_between
    day_offsets = {"$range": [0, day_count]}
    dates = {"$map": {
        "input": day_offsets,
        "as": "offset",
        "in": {"$dateAdd": {"startDate": start_date, "unit": "day", "amount": "$$offset"}},
    }}
    return {"$cond": [
        beyond_period,
        0,
        {"$let": {
            "vars": {"effective_start": effective_start},
            "in": {"$size": {"$filter": {
                "input": dates,
                "as": "day",
                "cond": {"$not": [{"$in": [{"$dayOfWeek": "$$day"}, [1, 7]]}]},
            }}},
        }},
    ]}


def present_days_expression(logs):
    qualifying = {"$filter": {
        "input": logs,
        "as": "log",
        "cond": {"$and": [
            {"$in": ["$$log.status", ["PRESENT", "WFH", "ON_DUTY"]]},
            {"$not": [{"$in": [{"$dayOfWeek": {"$dateFromString": {
                "dateString": "$$log.date", "format": "%Y-%m-%d",
            }}}, [1, 7]]}]},
        ]},
    }}
    return {"$sum": {"$map": {
        "input": qualifying,
        "as": "log",
        "in": {"$cond": [{"$eq": [{"$ifNull": ["$$log.half_day", False]}, True]}, 0.5, 1]},
    }}}


def late_logs_expression(logs):
    return {"$filter": {
        "input": logs,
        "as": "log",
        "cond": {"$gt": [{"$ifNull": ["$$log.late_minutes", 0]}, 0]},
    }}


def trend_date_count(from_date: str, to_date: str) -> int:
    try:
        start_day = date.fromisoformat(from_date)
    except ValueError:
        raise RequestValidationError([{
            "type": "value_error", "loc": ("query", "from"),
            "msg": "from must be a valid calendar date", "input": from_date,
        }]) from None
    try:
        end_day = date.fromisoformat(to_date)
    except ValueError:
        raise RequestValidationError([{
            "type": "value_error", "loc": ("query", "to"),
            "msg": "to must be a valid calendar date", "input": to_date,
        }]) from None
    days = (end_day - start_day).days + 1
    if days < 1:
        raise RequestValidationError([{
            "type": "value_error", "loc": ("query", "to"),
            "msg": "to must be on or after from", "input": to_date,
        }])
    if days > 92:
        raise RequestValidationError([{
            "type": "value_error", "loc": ("query", "to"),
            "msg": "date range must not exceed 92 inclusive days", "input": to_date,
        }])
    return days


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
CalendarDate = Annotated[
    str,
    StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$"),
    AfterValidator(valid_calendar_date),
    Field(json_schema_extra={"format": "date"}),
]
Month = Annotated[
    str,
    StringConstraints(pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    AfterValidator(valid_month),
]
ShiftTime = Annotated[str, StringConstraints(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")]
EpochMillis = Annotated[
    int,
    Field(strict=True, ge=100000000000, le=4102444800000, json_schema_extra={"format": "int64"}),
]
EpochMillisNullable = Annotated[
    EpochMillis | None,
    Field(json_schema_extra={"nullable": True}),
]
Status = Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]
PresenceStatus = Literal["PRESENT", "WFH", "ON_DUTY"]
Page = Annotated[int, Query(ge=1)]
PageSize = Annotated[int, Query(ge=1, le=100)]


class EmployeeCreate(BaseModel):
    emp_code: str = Field(
        pattern=r"^EMP\d{4,6}$",
        description="The customer's own code, supplied by the client; unique and immutable.",
        examples=["EMP0001"],
    )
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: ShiftTime = Field(default="09:30", description="24-hour HH:MM, IST.")
    shift_end: ShiftTime = Field(default="18:30", description="24-hour HH:MM, IST; <= shift_start means overnight.")
    joined_on: CalendarDate = Field(description="First day of employment, YYYY-MM-DD.")

    @model_validator(mode="after")
    def different_shift_times(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start and shift_end must differ")
        return self


class Employee(BaseModel):
    emp_code: str
    name: str
    email: str
    department: str
    shift_start: str
    shift_end: str
    joined_on: CalendarDate
    created_at: EpochMillis = Field(description="Server-set creation time, epoch milliseconds.", json_schema_extra={"readOnly": True})


class EmployeePage(BaseModel):
    items: list[Employee]
    total: int
    page: int
    page_size: int


class PunchInRequest(BaseModel):
    emp_code: str
    punched_at: EpochMillis = Field(default_factory=now_millis, description="Epoch milliseconds. Defaults to now.")
    status: PresenceStatus = "PRESENT"


class PunchOutRequest(BaseModel):
    emp_code: str
    punched_at: EpochMillis = Field(default_factory=now_millis, description="Epoch milliseconds. Defaults to now.")


class RegularizeRequest(BaseModel):
    status: Status = Field(default_factory=lambda: None)
    punch_in: EpochMillis = Field(default_factory=lambda: None)
    punch_out: EpochMillis = Field(default_factory=lambda: None)
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)


class HistoryChange(BaseModel):
    from_value: Any = Field(default_factory=lambda: None, alias="from", json_schema_extra={"nullable": True})
    to_value: Any = Field(default_factory=lambda: None, alias="to", json_schema_extra={"nullable": True})


class HistoryEntry(BaseModel):
    at: EpochMillis
    by: str
    reason: str
    changes: dict[str, HistoryChange]


class AttendanceRecord(BaseModel):
    emp_code: str
    date: CalendarDate = Field(description="The attendance day (R1), YYYY-MM-DD.")
    status: Status
    punch_in: EpochMillisNullable
    punch_out: EpochMillisNullable
    work_hours: float | None = Field(description="Computed (R4). Null until punch-out.", json_schema_extra={"readOnly": True})
    late_minutes: int = Field(json_schema_extra={"readOnly": True})
    overtime_minutes: int = Field(json_schema_extra={"readOnly": True})
    half_day: bool = Field(json_schema_extra={"readOnly": True})
    history: list[HistoryEntry]


class AttendancePage(BaseModel):
    items: list[AttendanceRecord]
    total: int
    page: int
    page_size: int


class EmployeeMonthly(BaseModel):
    emp_code: str
    month: str
    working_days: int
    present_days: float
    leave_days: int
    late_count: int
    total_late_minutes: int
    total_overtime_minutes: int
    attendance_pct: float | None = Field(json_schema_extra={"nullable": True})


class DepartmentSummaryItem(BaseModel):
    department: str
    headcount: int
    present_days: float
    avg_work_hours: float | None = Field(json_schema_extra={"nullable": True})
    late_count: int
    total_late_minutes: int
    leave_count: int
    on_duty_count: int


class DepartmentSummary(BaseModel):
    month: str
    items: list[DepartmentSummaryItem]


class LeaderboardItem(BaseModel):
    rank: int = Field(ge=1)
    emp_code: str
    name: str
    department: str
    total_late_minutes: int
    late_count: int


class Leaderboard(BaseModel):
    month: str
    items: list[LeaderboardItem]


class TrendItem(BaseModel):
    date: CalendarDate
    is_working_day: bool
    headcount: int
    present_count: float
    late_count: int
    attendance_rate: float | None = Field(json_schema_extra={"nullable": True})
    moving_avg_7d: float | None = Field(json_schema_extra={"nullable": True})


class Trend(BaseModel):
    department: str
    items: list[TrendItem]


class HealthResponse(BaseModel):
    status: Literal["ok"]


ExplainEndpoint = Literal[
    "attendance_list", "employee_monthly", "department_summary",
    "late_leaderboard", "department_trend",
]


class ExplainResponse(BaseModel):
    endpoint: str
    collection: str
    explain: dict[str, Any]


class ErrorResponse(BaseModel):
    detail: str


class ValidationErrorResponse(BaseModel):
    detail: list[dict[str, Any]]


ERROR_RESPONSE = {"model": ErrorResponse, "description": "Error"}
VALIDATION_RESPONSE = {
    "model": ValidationErrorResponse,
    "description": "Validation error (FastAPI default body)",
}


# --------------------------------------------------------------------------- #
# Existing endpoints 
# --------------------------------------------------------------------------- #
@app.get(
    "/health",
    operation_id="health",
    response_model=HealthResponse,
    responses={503: ERROR_RESPONSE},
    summary="Readiness probe",
    description="Returns 200 only if MongoDB answers a ping, otherwise 503.",
)
async def health(request: Request):
    try:
        async with asyncio.timeout(5):
            await request.app.state.db.command("ping")
    except (PyMongoError, TimeoutError):
        raise HTTPException(503, "MongoDB is unavailable") from None
    return {"status": "ok"}


@app.post(
    "/employees",
    status_code=201,
    operation_id="createEmployee",
    response_model=Employee,
    responses={409: ERROR_RESPONSE, 422: VALIDATION_RESPONSE},
    summary="Create an employee",
    description="The client supplies a unique emp_code. Shift start and end must differ; created_at is set by the server.",
)
async def create_employee(body: EmployeeCreate, request: Request):
    doc = body.model_dump()
    doc["created_at"] = utc_seconds(datetime.now(UTC))
    try:
        await request.app.state.db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists") from None
    return serialize(doc)


@app.get(
    "/employees",
    operation_id="listEmployees",
    response_model=EmployeePage,
    responses={422: VALIDATION_RESPONSE},
    summary="List employees",
    description="Sorted by emp_code ascending. total counts the filtered set.",
)
async def list_employees(
    request: Request,
    department: str = Query(default=None),
    page: Page = 1,
    page_size: PageSize = 20,
):
    query = {} if department is None else {"department": department}
    index = "employee_code_unique" if department is None else "department_employee"
    collection = request.app.state.db.employees
    cursor = collection.find(query, {"_id": 0}).hint(index).sort("emp_code", 1)
    cursor = cursor.skip((page - 1) * page_size).limit(page_size)
    total, items = await asyncio.gather(
        collection.count_documents(query, hint=index), cursor.to_list(length=page_size),
    )
    return {"items": serialize(items), "total": total, "page": page, "page_size": page_size}


@app.post(
    "/attendance/punch-in",
    status_code=201,
    operation_id="punchIn",
    response_model=AttendanceRecord,
    responses={404: ERROR_RESPONSE, 409: ERROR_RESPONSE, 422: VALIDATION_RESPONSE},
    summary="Record a punch-in",
    description="Creates the day's record using R1 and R2. Omitted punched_at defaults to now; status defaults to PRESENT.",
)
async def punch_in(body: PunchInRequest, request: Request):
    db = request.app.state.db
    emp = await db.employees.find_one({"emp_code": body.emp_code})
    if emp is None:
        raise HTTPException(404, "employee not found")
    ts = from_epoch_millis(body.punched_at)
    day = attendance_date(ts, emp["shift_start"], emp["shift_end"])
    doc = {
        "emp_code": body.emp_code,
        "date": day,
        "status": body.status,
        "punch_in": ts,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": compute_late_minutes(ts, emp["shift_start"], day),
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }
    try:
        await db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date") from None
    return attendance_response(doc)


@app.get(
    "/attendance",
    operation_id="listAttendance",
    response_model=AttendancePage,
    responses={422: VALIDATION_RESPONSE},
    summary="List attendance records",
    description="Sorted by date descending, then emp_code ascending. Date bounds are inclusive; date_from must not exceed date_to.",
)
async def list_attendance(
    request: Request,
    emp_code: str = Query(default=None),
    date_from: CalendarDate = None,
    date_to: CalendarDate = None,
    status: Status = Query(default=None),
    page: Page = 1,
    page_size: PageSize = 20,
):
    operation = attendance_list_operation(emp_code, date_from, date_to, status, page, page_size)
    query, index = operation["filter"], operation["hint"]
    collection = request.app.state.db.attendance_logs
    cursor = collection.find(query, {"_id": 0}).hint(index).sort(list(operation["sort"].items()))
    cursor = cursor.skip(operation["skip"]).limit(operation["limit"])
    total, items = await asyncio.gather(
        collection.count_documents(query, hint=index), cursor.to_list(length=page_size),
    )
    return {"items": [attendance_response(doc) for doc in items], "total": total, "page": page, "page_size": page_size}


def body_validation_error(message: str, field: str | None = None):
    return RequestValidationError([{
        "type": "value_error",
        "loc": ("body", field) if field is not None else ("body",),
        "msg": message,
    }])


def attendance_snapshot_query(doc):
    """Match the mutable state used to calculate a conditional attendance update."""
    query = {
        "_id": doc["_id"],
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": doc.get("punch_in"),
        "punch_out": doc.get("punch_out"),
    }
    if "history" in doc:
        query["history"] = doc["history"]
    else:
        query["history"] = {"$exists": False}
    return query


@app.post(
    "/attendance/punch-out",
    operation_id="punchOut",
    response_model=AttendanceRecord,
    responses={404: ERROR_RESPONSE, 409: ERROR_RESPONSE, 422: VALIDATION_RESPONSE},
    summary="Record a punch-out",
    description="Closes the latest record whose punch_in is at or before punched_at and computes R3-R5. An earlier timestamp than every punch-in is a 422 validation error.",
)
async def punch_out(body: PunchOutRequest, request: Request):
    db = request.app.state.db
    employee = await db.employees.find_one({"emp_code": body.emp_code})
    if employee is None:
        raise HTTPException(404, "employee not found")

    punched_at = from_epoch_millis(body.punched_at)
    record = await db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": punched_at}},
        sort=[("punch_in", -1)],
    )
    if record is None:
        # Keep the eligible-record lookup; distinguish an early time from no punch-in.
        later_record = await db.attendance_logs.find_one(
            {"emp_code": body.emp_code, "punch_in": {"$gt": punched_at}}, {"_id": 1},
        )
        if later_record is not None:
            raise body_validation_error("punched_at must be after punch_in", "punched_at")
        raise HTTPException(404, "no punch-in found")
    if record.get("punch_out") is not None:
        raise HTTPException(409, "attendance record is already punched out")

    punch_in = utc_seconds(record["punch_in"])
    punched_at = utc_seconds(punched_at)
    duration = punched_at - punch_in
    if duration <= timedelta(0) or duration > timedelta(hours=24):
        raise body_validation_error("punched_at must be after punch_in and within 24 hours", "punched_at")

    updates = {
        "punch_out": punched_at,
        "work_hours": compute_work_hours(punch_in, punched_at),
        "overtime_minutes": compute_overtime(
            punched_at, employee["shift_start"], employee["shift_end"], record["date"]
        ),
    }
    updates["half_day"] = updates["work_hours"] < 4.50
    updated = await db.attendance_logs.find_one_and_update(
        attendance_snapshot_query(record),
        {"$set": updates},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, "attendance record changed concurrently")
    return attendance_response(updated)


@app.patch(
    "/attendance/{emp_code}/{date}",
    operation_id="regularizeAttendance",
    response_model=AttendanceRecord,
    responses={404: ERROR_RESPONSE, 409: ERROR_RESPONSE, 422: VALIDATION_RESPONSE},
    summary="Regularize (manually correct) one day's record",
    description="Omitted fields are unchanged. Recomputes derived fields and appends one history entry with actual changes. ABSENT/LEAVE clear times; presence requires punch_in; no-op corrections are rejected.",
)
async def regularize_attendance(emp_code: str, date: CalendarDate, body: RegularizeRequest, request: Request):
    db = request.app.state.db
    employee = await db.employees.find_one({"emp_code": emp_code})
    if employee is None:
        raise HTTPException(404, "employee not found")

    record = await db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if record is None:
        raise HTTPException(404, "attendance record not found")

    supplied = body.model_fields_set
    status = body.status if "status" in supplied else record["status"]
    if status in ("ABSENT", "LEAVE"):
        if "punch_in" in supplied or "punch_out" in supplied:
            raise body_validation_error("ABSENT and LEAVE cannot include punch times")
        punch_in = punch_out = None
    else:
        punch_in = from_epoch_millis(body.punch_in) if "punch_in" in supplied else record.get("punch_in")
        punch_out = from_epoch_millis(body.punch_out) if "punch_out" in supplied else record.get("punch_out")
        if punch_in is None:
            raise body_validation_error("presence status requires punch_in", "punch_in")
        punch_in = utc_seconds(punch_in)
        if attendance_date(punch_in, employee["shift_start"], employee["shift_end"]) != date:
            raise body_validation_error("punch_in must belong to the attendance date", "punch_in")
        if punch_out is not None:
            punch_out = utc_seconds(punch_out)
            duration = punch_out - punch_in
            if duration <= timedelta(0) or duration > timedelta(hours=24):
                raise body_validation_error("punch_out must be after punch_in and within 24 hours", "punch_out")

    if punch_in is not None:
        punch_in = utc_seconds(punch_in)
    if punch_out is not None:
        punch_out = utc_seconds(punch_out)

    changes = {}
    old_values = {
        "status": record["status"],
        "punch_in": record.get("punch_in"),
        "punch_out": record.get("punch_out"),
        "work_hours": record.get("work_hours"),
        "late_minutes": record.get("late_minutes", 0),
        "overtime_minutes": record.get("overtime_minutes", 0),
        "half_day": record.get("half_day", False),
    }
    if status in ("ABSENT", "LEAVE"):
        derived = {"work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False}
    else:
        work_hours = compute_work_hours(punch_in, punch_out) if punch_out is not None else None
        derived = {
            "work_hours": work_hours,
            "late_minutes": compute_late_minutes(punch_in, employee["shift_start"], date),
            "overtime_minutes": compute_overtime(
                punch_out, employee["shift_start"], employee["shift_end"], date
            ) if punch_out is not None else 0,
            "half_day": work_hours < 4.50 if work_hours is not None else False,
        }
    new_values = {
        "status": status,
        "punch_in": punch_in,
        "punch_out": punch_out,
        **derived,
    }
    for field, new_value in new_values.items():
        old_value = old_values[field]
        if isinstance(old_value, datetime):
            old_value = utc_seconds(old_value)
        if old_value != new_value:
            changes[field] = {"from": old_value, "to": new_value}
    if not changes:
        raise body_validation_error("regularization does not change the record")

    now = utc_seconds(datetime.now(UTC))
    history_entry = {
        "at": now,
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": changes,
    }
    updated = await db.attendance_logs.find_one_and_update(
        attendance_snapshot_query(record),
        {"$set": new_values, "$push": {"history": history_entry}},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, "attendance record changed concurrently")
    return attendance_response(updated)


# --------------------------------------------------------------------------- #
# Analytics
# --------------------------------------------------------------------------- #
def employee_monthly_pipeline(emp_code: str, month: str):
    month_start, month_end = month_bounds(month)
    pipeline = [
        {"$match": {"emp_code": emp_code}},
        {"$set": {"working_days": working_days_expression("$joined_on", month_start, month_end)}},
        {"$lookup": {
            "from": "attendance_logs",
            "let": {"code": "$emp_code"},
            "pipeline": [
                {"$match": {"$expr": {"$and": [
                    {"$eq": ["$emp_code", "$$code"]},
                    {"$gte": ["$date", month_start]},
                    {"$lt": ["$date", month_end]},
                ]}}},
                {"$project": {
                    "_id": 0, "date": 1, "status": 1, "half_day": 1,
                    "late_minutes": 1, "overtime_minutes": 1,
                }},
            ],
            "as": "_logs",
        }},
        {"$set": {
            "present_days": present_days_expression("$_logs"),
            "leave_days": {"$size": {"$filter": {
                "input": "$_logs", "as": "log", "cond": {"$eq": ["$$log.status", "LEAVE"]},
            }}},
            "_late_logs": late_logs_expression("$_logs"),
            "total_overtime_minutes": {"$sum": {"$map": {
                "input": "$_logs", "as": "log", "in": {"$ifNull": ["$$log.overtime_minutes", 0]},
            }}},
        }},
        {"$set": {
            "late_count": {"$size": "$_late_logs"},
            "total_late_minutes": {"$sum": {"$map": {
                "input": "$_late_logs", "as": "log", "in": {"$ifNull": ["$$log.late_minutes", 0]},
            }}},
            "attendance_pct": {"$cond": [
                {"$gt": ["$working_days", 0]},
                round_half_up_expression({"$multiply": [
                    {"$divide": ["$present_days", "$working_days"]}, 100,
                ]}, 4),
                None,
            ]},
        }},
        {"$project": {
            "_id": 0, "emp_code": 1, "month": {"$literal": month},
            "working_days": 1, "present_days": 1, "leave_days": 1, "late_count": 1,
            "total_late_minutes": 1, "total_overtime_minutes": 1, "attendance_pct": 1,
        }},
    ]
    return pipeline


def department_summary_pipeline(month: str, department: str | None):
    month_start, month_end = month_bounds(month)
    employee_filter = {"joined_on": {"$lt": month_end}}
    if department is not None:
        employee_filter["department"] = department
    return [
        {"$match": employee_filter},
        {"$lookup": {
            "from": "attendance_logs",
            "let": {"code": "$emp_code"},
            "pipeline": [
                {"$match": {"$expr": {"$and": [
                    {"$eq": ["$emp_code", "$$code"]},
                    {"$gte": ["$date", month_start]},
                    {"$lt": ["$date", month_end]},
                ]}}},
                {"$project": {
                    "_id": 0, "date": 1, "status": 1, "half_day": 1,
                    "work_hours": 1, "late_minutes": 1,
                }},
            ],
            "as": "_logs",
        }},
        {"$set": {
            "_present_days": present_days_expression("$_logs"),
            "_late_logs": late_logs_expression("$_logs"),
            "_work_logs": {"$filter": {
                "input": "$_logs", "as": "log", "cond": {"$and": [
                    {"$in": ["$$log.status", ["PRESENT", "WFH", "ON_DUTY"]]},
                    {"$ne": [{"$ifNull": ["$$log.work_hours", None]}, None]},
                ]},
            }},
            "_leave_count": {"$size": {"$filter": {
                "input": "$_logs", "as": "log", "cond": {"$eq": ["$$log.status", "LEAVE"]},
            }}},
            "_on_duty_count": {"$size": {"$filter": {
                "input": "$_logs", "as": "log", "cond": {"$eq": ["$$log.status", "ON_DUTY"]},
            }}},
        }},
        {"$group": {
            "_id": "$department",
            "headcount": {"$sum": 1},
            "present_days": {"$sum": "$_present_days"},
            "_late_count": {"$sum": {"$size": "$_late_logs"}},
            "_total_late_minutes": {"$sum": {"$sum": {"$map": {
                "input": "$_late_logs", "as": "log", "in": {"$ifNull": ["$$log.late_minutes", 0]},
            }}}},
            "leave_count": {"$sum": "$_leave_count"},
            "on_duty_count": {"$sum": "$_on_duty_count"},
            "_work_hours_total": {"$sum": {"$sum": {"$map": {
                "input": "$_work_logs", "as": "log", "in": "$$log.work_hours",
            }}}},
            "_work_hours_count": {"$sum": {"$size": "$_work_logs"}},
        }},
        {"$project": {
            "_id": 0, "department": "$_id", "headcount": 1, "present_days": 1,
            "avg_work_hours": {"$cond": [
                {"$gt": ["$_work_hours_count", 0]},
                round_half_up_expression({"$divide": ["$_work_hours_total", "$_work_hours_count"]}, 2),
                None,
            ]},
            "late_count": "$_late_count", "total_late_minutes": "$_total_late_minutes",
            "leave_count": 1, "on_duty_count": 1,
        }},
        {"$sort": {"department": 1}},
    ]


def late_leaderboard_pipeline(month: str, limit: int, department: str | None):
    month_start, month_end = month_bounds(month)
    employee_lookup = [
        {"$match": {"$expr": {"$eq": ["$emp_code", "$$code"]}}},
        {"$project": {"_id": 0, "emp_code": 1, "name": 1, "department": 1}},
    ]
    pipeline = [
        {"$match": {
            "date": {"$gte": month_start, "$lt": month_end},
            "late_minutes": {"$gt": 0},
        }},
        {"$group": {
            "_id": "$emp_code",
            "total_late_minutes": {"$sum": "$late_minutes"},
            "late_count": {"$sum": 1},
        }},
        {"$lookup": {"from": "employees", "let": {"code": "$_id"}, "pipeline": employee_lookup, "as": "_employee"}},
        {"$unwind": "$_employee"},
    ]
    if department is not None:
        pipeline.append({"$match": {"_employee.department": department}})
    pipeline.extend([
        {"$setWindowFields": {"sortBy": {"total_late_minutes": -1}, "output": {"rank": {"$rank": {}}}}},
        {"$match": {"rank": {"$lte": limit}}},
        {"$sort": {"total_late_minutes": -1, "_employee.emp_code": 1}},
        {"$project": {
            "_id": 0, "rank": 1, "emp_code": "$_employee.emp_code", "name": "$_employee.name",
            "department": "$_employee.department", "total_late_minutes": 1, "late_count": 1,
        }},
    ])
    return pipeline


def department_trend_pipeline(department: str, from_date: str, days: int):
    start_day = date.fromisoformat(from_date)
    start_datetime = datetime.combine(start_day, time.min, tzinfo=UTC)
    return [
        {"$match": {"department": department}},
        {"$limit": 1},
        {"$lookup": {
            "from": "employees", "let": {"department": "$department"},
            "pipeline": [
                {"$match": {"$expr": {"$eq": ["$department", "$$department"]}}},
                {"$project": {"_id": 0, "emp_code": 1, "joined_on": 1}},
            ],
            "as": "_employees",
        }},
        {"$set": {
            "_employee_codes": {"$map": {"input": "$_employees", "as": "employee", "in": "$$employee.emp_code"}},
            "_days": {"$map": {
                "input": {"$range": [0, days]}, "as": "offset",
                "in": {"$dateAdd": {"startDate": start_datetime, "unit": "day", "amount": "$$offset"}},
            }},
        }},
        {"$unwind": "$_days"},
        {"$set": {
            "_date": "$_days",
            "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$_days", "timezone": "UTC"}},
            "is_working_day": {"$not": [{"$in": [{"$dayOfWeek": "$_days"}, [1, 7]]}]},
            "headcount": {"$size": {"$filter": {
                "input": "$_employees", "as": "employee",
                "cond": {"$lte": ["$$employee.joined_on", {"$dateToString": {
                    "format": "%Y-%m-%d", "date": "$_days", "timezone": "UTC",
                }}]},
            }}},
        }},
        {"$lookup": {"from": "attendance_logs", "localField": "date", "foreignField": "date", "as": "_all_day_logs"}},
        {"$set": {"_day_logs": {"$filter": {
            "input": "$_all_day_logs", "as": "log", "cond": {"$in": ["$$log.emp_code", "$_employee_codes"]},
        }}}},
        {"$set": {
            "_present_logs": {"$filter": {
                "input": "$_day_logs", "as": "log", "cond": {"$in": ["$$log.status", ["PRESENT", "WFH", "ON_DUTY"]]},
            }},
            "_late_logs": late_logs_expression("$_day_logs"),
        }},
        {"$set": {
            "present_count": {"$sum": {"$map": {
                "input": "$_present_logs", "as": "log",
                "in": {"$cond": [{"$eq": [{"$ifNull": ["$$log.half_day", False]}, True]}, 0.5, 1]},
            }}},
            "late_count": {"$size": "$_late_logs"},
        }},
        {"$set": {"attendance_rate": {"$cond": [
            {"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]},
            round_half_up_expression({"$divide": ["$present_count", "$headcount"]}, 4), None,
        ]}}},
        {"$setWindowFields": {"sortBy": {"_date": 1}, "output": {
            "_moving_avg_raw": {"$avg": "$attendance_rate", "window": {"documents": [-6, 0]}},
        }}},
        {"$set": {"moving_avg_7d": {"$cond": [
            {"$eq": [{"$ifNull": ["$_moving_avg_raw", None]}, None]}, None,
            round_half_up_expression("$_moving_avg_raw", 4),
        ]}}},
        {"$sort": {"_date": 1}},
        {"$project": {
            "_id": 0, "date": 1, "is_working_day": 1, "headcount": 1,
            "present_count": 1, "late_count": 1, "attendance_rate": 1, "moving_avg_7d": 1,
        }},
    ]


@app.get(
    "/analytics/employees/{emp_code}/monthly",
    operation_id="employeeMonthly",
    response_model=EmployeeMonthly,
    responses={404: ERROR_RESPONSE, 422: VALIDATION_RESPONSE},
    summary="Monthly attendance summary for one employee",
    description=(
        "MongoDB aggregation using R5-R8 and stored derived attendance values. "
        "Existing employees with no records receive a zero-valued summary."
    ),
)
async def employee_monthly(emp_code: str, month: Month, request: Request):
    pipeline = employee_monthly_pipeline(emp_code, month)
    cursor = await request.app.state.db.employees.aggregate(pipeline, hint="employee_code_unique")
    result = await cursor.to_list(length=1)
    if not result:
        raise HTTPException(404, "employee not found")
    return result[0]


@app.get(
    "/analytics/departments/summary",
    operation_id="departmentSummary",
    response_model=DepartmentSummary,
    responses={422: VALIDATION_RESPONSE},
    summary="Department summary for a month",
    description="MongoDB aggregation across employees and attendance logs using R5-R9 and stored derived values.",
)
async def department_summary(
    request: Request,
    month: Month,
    department: str = Query(default=None),
):
    pipeline = department_summary_pipeline(month, department)
    cursor = await request.app.state.db.employees.aggregate(pipeline, hint="employee_joined_on")
    items = await cursor.to_list(length=None)
    return {"month": month, "items": items}


@app.get(
    "/analytics/leaderboard/late",
    operation_id="lateLeaderboard",
    response_model=Leaderboard,
    responses={422: VALIDATION_RESPONSE},
    summary="Late-comers leaderboard for a month",
    description="Ranks employees by stored late minutes using MongoDB competition ranking; the limit is applied after ranking.",
)
async def late_leaderboard(
    request: Request,
    month: Month,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    department: str = Query(default=None),
):
    pipeline = late_leaderboard_pipeline(month, limit, department)
    cursor = await request.app.state.db.attendance_logs.aggregate(pipeline, hint="date_employee")
    items = await cursor.to_list(length=None)
    return {"month": month, "items": items}


@app.get(
    "/analytics/departments/{department}/trend",
    operation_id="departmentTrend",
    response_model=Trend,
    responses={404: ERROR_RESPONSE, 422: VALIDATION_RESPONSE},
    summary="Daily attendance trend for a department",
    description="Returns one MongoDB gap-filled row per date and a seven-row moving average; the inclusive range is at most 92 days.",
)
async def department_trend(
    request: Request,
    department: str,
    from_date: CalendarDate = Query(..., alias="from", json_schema_extra={"format": "date"}),
    to_date: CalendarDate = Query(..., alias="to", json_schema_extra={"format": "date"}),
):
    days = trend_date_count(from_date, to_date)
    pipeline = department_trend_pipeline(department, from_date, days)
    cursor = await request.app.state.db.employees.aggregate(pipeline, hint="department_employee")
    items = await cursor.to_list(length=92)
    if not items:
        raise HTTPException(404, "department not found")
    return {"department": department, "items": items}


def explain_validation_error(parameter: str):
    return RequestValidationError([{
        "type": "missing",
        "loc": ("query", parameter),
        "msg": "Field required for this endpoint",
        "input": None,
    }])


def validate_explain_dates(month, *date_values):
    if month is not None:
        try:
            month_bounds(month)
        except ValueError:
            raise RequestValidationError([{
                "type": "value_error", "loc": ("query", "month"),
                "msg": "month must be a valid calendar month in YYYY-MM format", "input": month,
            }]) from None
    for parameter, value in date_values:
        if value is not None:
            try:
                date.fromisoformat(value)
            except ValueError:
                raise RequestValidationError([{
                    "type": "value_error", "loc": ("query", parameter),
                    "msg": f"{parameter} must be a valid calendar date", "input": value,
                }]) from None


@app.get(
    "/admin/explain/{endpoint}",
    operation_id="explainEndpoint",
    response_model=ExplainResponse,
    responses={422: VALIDATION_RESPONSE},
    summary="MongoDB explain for an endpoint's main query",
    description=(
        "Runs MongoDB explain with executionStats for the named endpoint's main query or aggregation, "
        "using the same filters, sort, pagination, and index hint as the endpoint. Returns the raw explain output."
    ),
)
async def explain_endpoint(
    endpoint: ExplainEndpoint,
    request: Request,
    emp_code: str = Query(default=None),
    month: str = Query(default=None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: str = Query(default=None),
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    date_from: str = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$", json_schema_extra={"format": "date"}),
    date_to: str = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$", json_schema_extra={"format": "date"}),
    status: Status = Query(default=None),
    from_date: str = Query(default=None, alias="from", pattern=r"^\d{4}-\d{2}-\d{2}$", json_schema_extra={"format": "date"}),
    to_date: str = Query(default=None, alias="to", pattern=r"^\d{4}-\d{2}-\d{2}$", json_schema_extra={"format": "date"}),
    page: Page = 1,
    page_size: PageSize = 20,
):
    validate_explain_dates(month, ("date_from", date_from), ("date_to", date_to), ("from", from_date), ("to", to_date))
    if endpoint == "attendance_list":
        operation = attendance_list_operation(emp_code, date_from, date_to, status, page, page_size)
        command = {
            "find": "attendance_logs",
            "filter": operation["filter"],
            "projection": {"_id": 0},
            "sort": operation["sort"],
            "skip": operation["skip"],
            "limit": operation["limit"],
            "hint": operation["hint"],
        }
        collection = "attendance_logs"
    elif endpoint == "employee_monthly":
        if emp_code is None:
            raise explain_validation_error("emp_code")
        if month is None:
            raise explain_validation_error("month")
        command = {
            "aggregate": "employees",
            "pipeline": employee_monthly_pipeline(emp_code, month),
            "cursor": {},
            "hint": "employee_code_unique",
        }
        collection = "employees"
    elif endpoint == "department_summary":
        if month is None:
            raise explain_validation_error("month")
        command = {
            "aggregate": "employees",
            "pipeline": department_summary_pipeline(month, department),
            "cursor": {},
            "hint": "employee_joined_on",
        }
        collection = "employees"
    elif endpoint == "late_leaderboard":
        if month is None:
            raise explain_validation_error("month")
        command = {
            "aggregate": "attendance_logs",
            "pipeline": late_leaderboard_pipeline(month, limit, department),
            "cursor": {},
            "hint": "date_employee",
        }
        collection = "attendance_logs"
    else:
        if department is None:
            raise explain_validation_error("department")
        if from_date is None:
            raise explain_validation_error("from")
        if to_date is None:
            raise explain_validation_error("to")
        days = trend_date_count(from_date, to_date)
        command = {
            "aggregate": "employees",
            "pipeline": department_trend_pipeline(department, from_date, days),
            "cursor": {},
            "hint": "department_employee",
        }
        collection = "employees"

    db = request.app.state.db
    explain = await db.command({
        "explain": command,
        "verbosity": "executionStats",
    })
    raw_json = json.loads(json_util.dumps(explain, json_options=json_util.RELAXED_JSON_OPTIONS))
    return {"endpoint": endpoint, "collection": collection, "explain": raw_json}

