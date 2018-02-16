#pragma once

#include "types.h"

namespace aten
{
    enum MeshVertexFormat : uint32_t {
        Position,
        Normal,
        Color,
        UV,
        Tangent,
        BlendIndices,
        BlendWeight,

		Num,
    };

    enum MeshVertexSize : uint32_t {
        Position     = sizeof(float) * 4, ///< 頂点位置
        Normal       = sizeof(float) * 3, ///< 法線
        Color        = sizeof(uint8_t) * 4,     ///< 頂点カラー
        UV           = sizeof(float) * 2, ///< UV座標
        Tangent      = sizeof(float) * 3, ///< 接ベクトル
		BlendIndices = sizeof(float) * 4, ///< ブレンドマトリクスインデックス
		BlendWeight  = sizeof(float) * 4, ///< ブレンドウエイト
    };

	enum {
		MaxJointMtxNum = 4,
	};

    // フォーマット
    // +------------------------+
    // |         ヘッダ     　　|
    // +------------------------+
    // |    メッシュグループ    |
    // +------------------------+

    // メッシュグループ
    // +------------------------+
    // |     グループヘッダ     |
    // +------------------------+
    // |   頂点データテーブル   |
    // | +--------------------+ |
    // | |      ヘッダ        | |
    // | +--------------------+ |
    // | |     頂点データ     | |
    // | +--------------------+ |
    // |         ・・・         |
    // +------------------------+
    // |    メッシュテーブル    |
    // | +--------------------+ |
    // | |      メッシュ      | |
    // | |+------------------+| |
    // | ||     ヘッダ       || |
    // | |+------------------+| |
    // | |                    | |
    // | |     サブセット     | |
    // | |+------------------+| |
    // | ||     ヘッダ       || |
    // | |+------------------+| |
    // | ||インデックスデータ|| |
    // | |+------------------+| |
    // | |      ・・・        | |
    // | +--------------------+ |
    // |        ・・・          |
    // +------------------------+
    
    struct MeshHeader {
        uint32_t magic;
        uint32_t version;

        uint32_t sizeHeader;
        uint32_t sizeFile;

        float maxVtx[3];
        float minVtx[3];

        uint16_t numVB;
        uint16_t numMeshGroup;
        uint16_t numMeshSet;
        uint16_t numMeshSubset;

        uint32_t numAllJointIndices; ///< ジョイントインデックス総数
    };

    /////////////////////////////////////////////////////////

    // マテリアル情報
    struct MeshMaterial {
        char name[32];  ///< マテリアル名
        uint32_t nameKey;    ///< マテリアル名キー
    };

    /////////////////////////////////////////////////////////

    // 頂点データ情報
    struct MeshVertex {
        uint16_t sizeVtx;  ///< １頂点あたりのサイズ
        uint16_t numVtx;   ///< 頂点数
    };

    // メッシュグループ情報
    struct MeshGroup {
        uint16_t numVB;        ///< 頂点バッファ数
        uint16_t numMeshSet;   ///< メッシュセット数
    };

    // メッシュセット情報
    struct MeshSet {
        uint16_t numSubset;

        uint16_t fmt;      ///< 頂点フォーマット

        float center[3]; ///< 重心位置

        MeshMaterial mtrl;    ///< マテリアル情報
    };

    // プリミティブセット情報
    struct PrimitiveSet {
        uint16_t idxVB;        ///< 利用する頂点バッファのインデックス
        uint16_t minIdx;
        uint16_t maxIdx;
        uint16_t numJoints;    ///< ジョイント数

        uint32_t numIdx;       ///< インデックス数
    };
}   // namespace izanagi
