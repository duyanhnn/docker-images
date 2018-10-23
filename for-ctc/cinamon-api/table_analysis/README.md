## Table analysis
Use result of line-cut from Bprost to process following tasks

**1. Line-cut expending**

- Use line-cut result of brpost, perform expend box of lines which were cut not enough information.

- Call `expend_line_cut` method to do task. View example in file `line_cut_expending.py` 

- Output is a file which contain dictionary with format: 
    ```
    {
    'line_1': {'location': [y1, x1, y2, x2]},
    ...
    }
    ```

**2. Table analysis**

- Use cell-cut result of bprost, determine talbe of cell, position (row, column) of cell in table and structure of tables

- Call `get_table_cell_information` method to do task, View example in file `table_analysis.py`

- Output `cells` and `tables` with format:
    
    **cells:** 
    ```
   {
    'cell1': {'table_name': 'table_1', 'position':[(row,column),...]},
    ...
    }
    ```
    
    **tables:**
    ```
    {
    'table_1': [['cell1', 'cell2', 'cell3', 'cell4'],
                ['cell5', 'cell6', 'cell7', 'cell8']],
    ...
    }
    ```