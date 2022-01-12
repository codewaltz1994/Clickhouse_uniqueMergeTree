#include <Storages/StorageBinary.h>
#include <Storages/StorageFactory.h>

#include <Interpreters/Context.h>
#include <Interpreters/evaluateConstantExpression.h>

#include <Parsers/ASTCreateQuery.h>
#include <Parsers/ASTLiteral.h>
#include <Parsers/ASTIdentifier.h>

#include <Columns/ColumnString.h>

#include <DataTypes/DataTypeString.h>
#include <DataTypes/DataTypesNumber.h>

#include <Common/escapeForFileName.h>
#include <Common/typeid_cast.h>
#include <Common/parseGlobs.h>
#include <Storages/ColumnsDescription.h>
#include <Storages/StorageInMemoryMetadata.h>

#include <Processors/Sources/SourceWithProgress.h>
#include <QueryPipeline/Pipe.h>

#include <rpc/client.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
    extern const int NOT_IMPLEMENTED;
    extern const int CANNOT_TRUNCATE_FILE;
    extern const int DATABASE_ACCESS_DENIED;
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int UNKNOWN_IDENTIFIER;
    extern const int INCORRECT_FILE_NAME;
    extern const int FILE_DOESNT_EXIST;
    extern const int TIMEOUT_EXCEEDED;
    extern const int INCOMPATIBLE_COLUMNS;
    extern const int CANNOT_STAT;
}

StorageBinary::StorageBinary(const StorageID & storage_id_, const std::string & user_files_path, UInt64 number_)
    : IStorage(storage_id_), file_path(user_files_path), number(number_)
{
    StorageInMemoryMetadata storage_metadata;
    storage_metadata.setColumns(
        ColumnsDescription({{"binary", std::make_shared<DataTypeString>()}, {"frame", std::make_shared<DataTypeUInt32>()}}));
    setInMemoryMetadata(storage_metadata);
}

class BinarySource : public SourceWithProgress
{
public:
    static constexpr size_t length = 2 * 1000 * 800 * 16;
    static Block getHeader(const StorageMetadataPtr & metadata_snapshot) { return metadata_snapshot->getSampleBlock(); }

    BinarySource(const StorageBinary & storage_, const StorageMetadataPtr & metadata_snapshot_, Names, UInt64 number)
        : SourceWithProgress(metadata_snapshot_->getSampleBlockNonMaterialized()), path(storage_.path()), max_number(number)
    {
    }

    String getName() const override { return "BinarySource"; }

    Chunk generate() override
    {
        if (current == max_number)
            return {};
        Columns columns;
        FILE * fp = fopen(path.c_str(), "rb");
        if (fp == nullptr)
            throw Exception("Open binary file failed.", ErrorCodes::BAD_ARGUMENTS);
        fseek(fp, 0, SEEK_SET);

        auto col = ColumnString::create();
        auto & data = col->getChars();
        auto & offset = col->getOffsets();

        data.resize(length + 1);
        fread(data.data(), sizeof(char), length, fp);

        offset.push_back(length + 1);
        columns.emplace_back(std::move(col));

        auto id_col = DataTypeUInt32{}.createColumnConst(1, current);
        columns.emplace_back(std::move(id_col));

        ++current;

        fclose(fp);

        return Chunk(std::move(columns), 1);
    }


private:
    UInt64 current = 0;
    std::string path;
    UInt64 max_number;
};


Pipe StorageBinary::read(
    const Names & column_names,
    const StorageMetadataPtr &  metadata_snapshot,
    SelectQueryInfo & /*query_info*/,
    ContextPtr, // context,
    QueryProcessingStage::Enum /*processed_stage*/,
    size_t, //max_block_size,
    unsigned) //num_streams)
{
    return Pipe(std::make_shared<BinarySource>(*this, metadata_snapshot, column_names, number));
}

void registerStorageBinary(StorageFactory & factory)
{
    factory.registerStorage("Binary", [](const StorageFactory::Arguments & factory_args) {
        ASTs & engine_args_ast = factory_args.engine_args;

        if (engine_args_ast.size() != 2) // NOLINT
            throw Exception(
                "StorageBinary requires 2 arguments: path of binary file and numbers.", ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

        engine_args_ast[0] = evaluateConstantExpressionOrIdentifierAsLiteral(engine_args_ast[0], factory_args.getLocalContext());
        engine_args_ast[1] = evaluateConstantExpressionOrIdentifierAsLiteral(engine_args_ast[1], factory_args.getLocalContext());
        auto path = engine_args_ast[0]->as<ASTLiteral &>().value.get<String>();
        auto number = engine_args_ast[1]->as<ASTLiteral &>().value.get<UInt64>();

        return StorageBinary::create(factory_args.table_id, path, number);
    });
}
}
