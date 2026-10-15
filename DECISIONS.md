# DECISIONS.md

1. **Indexes.** I think of indexes as shortcuts that help MongoDB find records.
   - **Employees:** `employee_code_unique` (`emp_code`) finds employees and stops duplicate codes. `department_employee` (`department, emp_code`) helps find employees by department. `employee_joined_on` (`joined_on`) helps check who had joined by the end of the month.
   - **Attendance:** `employee_date_unique` (`emp_code, date`) makes sure an employee has only one record per day. `date_employee` (`date, emp_code`) helps find records for a date or month. `status_date_employee` (`status, date, emp_code`) helps filter by status.
2. **Punch-in race.** If two identical punch-ins arrive together, both may try to save. The unique index allows one to save. It returns `201`; the other gets a duplicate-key error and returns `409`.
3. **Ties.** The leaderboard ranks people by late minutes. If several people tie at the limit, we return all of them, even if that means more rows than the limit.
4. **Headcount.** We start with the employees who count for that month, then look for their attendance. Employees with no records still count.
5. **100 times the data.** I would check which queries are slow. Then I would consider archiving old records or splitting the data across servers. I wouldn't increase the connection pool without checking the measurements.
