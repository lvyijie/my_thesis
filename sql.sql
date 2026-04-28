WITH CTE AS (
    SELECT *,
    ROW_NUMBER() OVER(PARTITION BY REPLACE(REPLACE(evaluation, ' ', ''), N'　', '') ORDER BY date) AS rn
FROM 豫西大峡谷
    )
INSERT INTO 豫西大峡谷简易去重(name, evaluation, date)
SELECT name, evaluation, date
FROM CTE
WHERE rn = 1;

WITH CTE AS (
    SELECT *,
    ROW_NUMBER() OVER(PARTITION BY REPLACE(REPLACE(evaluation, ' ', ''), N'　', '') ORDER BY date) AS rn
    FROM 豫西大峡谷字符清洗
    WHERE evaluation IS NOT NULL
        AND evaluation <> ''
        AND evaluation <> ' '
    )
INSERT INTO 豫西大峡谷清洗完毕 (name, evaluation, date)
SELECT name, evaluation, date
FROM CTE
WHERE rn = 1;