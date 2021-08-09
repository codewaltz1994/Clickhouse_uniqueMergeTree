## 1. 建表时指定缓存策略：

```sql
CREATE TABLE [IF NOT EXISTS] [db.]table_name [ON CLUSTER cluster]
(
    name1 [type1] [NULL|NOT NULL] [DEFAULT|MATERIALIZED|ALIAS expr1] [compression_codec] [TTL expr1],
    name2 [type2] [NULL|NOT NULL] [DEFAULT|MATERIALIZED|ALIAS expr2] [compression_codec] [TTL expr2],
    CACHE column_name cache_policy_name cluster_name(INTERVAL expr, INTERVAL expr),
    ...
) ENGINE = MergeTree()
ORDER BY expr
[PARTITION BY expr]
[PRIMARY KEY expr]
[SAMPLE BY expr]
[TTL expr
    [DELETE|TO DISK 'xxx'|TO VOLUME 'xxx' [, ...] ]
    [WHERE conditions]
    [GROUP BY key_expr [SET v1 = aggr_func(v1) [, v2 = aggr_func(v2) ...]] ] ]
[CACHE cache_policy_name cluster_name(INTERVAL expr, INTERVAL expr), ...]
[SETTINGS name=value, ...];
```

例：

```sql
CREATE TABLE t
(
    n Int32，
    s String,
    CACHE n FIFO cluster1(interval 1 hour, interval 4day),
    CACHE s CS cluster2(interval 4 hour, interval 1 day)
)ENGINE = MergeTree()
ORDER BY n
CACHE FIFO cluster1(interval 1 hour, interval 4 day), CS cluster2(interval 1 hour, interval 2 hour);
```

1. 实现ASTCache, ParserASTCache
2. 修改ASTCreateQuery, ParserCreateQuery
3. 实现CacheDescription, CachesDescription
4. 修改StorageInMemoryMetadata
5. 修改InterpreterCreateQuery
6. 修改StorageMergeTree
7. …

## 2. 修改表的CACHE策略

```sql
ALTER TABLE table_name ADD CACHE cache_policy_name cluster_name(INTERVAL expr, INTERVAL expr);
```

例：

```sql
ALTER TABLE t1 ADD CACHE FIFO cluster1(2 day, 4 day);
```

1. 增加AlterType ADD_CACHE
2. 修改ASTAlterCommand
3. 修改ASTAlterQuery
4. 修改InterpreterAlterQuery
5. 修改StorageMergeTree::alter
6. …

## 3. 删除表的缓存策略（不缓存）

```sql
ALTER TABLE table_name DROP CACHE cluster_name;
```

例：

```sql
ALTER TABLE t1 DROP CACHE cluster1;
```

1. 增加AlterType DROP_CACHE
2. 修改ASTAlterCommand
3. 修改ASTAlterQuery
4. 修改InterpreterAlterQuery
5. 修改StorageMergeTree::alter
6. …

## 4. 修改列的缓存策略

```sql
ALTER TABLE table_name MODIFY COLUMN CACHE column_name cache_policy_name cluster_name( INTERVAL expr, INTERVAL expr);
```

例：

```sql
ALTER TABLE t1 MODIFY COLUMN CACHE col1 FIFO cluster1 (1 day, 2day);
```

## 5. 删除列的缓存策略（不缓存）

```sql
ALTER TABLE table_name MODIFY COLUMN CACHE DROP column_name cluster_name;
```

例：

```sql
ALTER TABLE t1 MODIFY COLUMN CACHE DROP col1 cluster1;
```

和Column相关的Cache修改使用MODIFY_COLUMN类型

1. 修改ASTAlteCommand
2. 修改ASTAlterQuery
3. 修改InterpreterAlterQuery
4. 修改StorageMergeTree::alter
5. …
