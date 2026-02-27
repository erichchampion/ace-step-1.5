/**
 Pack two sequences by concatenating and sorting by mask (valid first). Mirrors Python pack_sequences.
 Returns (packed_hidden [B, L1+L2, D], new_mask [B, L1+L2]).
 */

import Foundation
import MLX

/// Pack hidden1 [B, L1, D] and hidden2 [B, L2, D] with masks [B, L1], [B, L2]. Valid (mask=1) positions come first per row.
public func packSequences(
    hidden1: MLXArray,
    hidden2: MLXArray,
    mask1: MLXArray,
    mask2: MLXArray
) -> (MLXArray, MLXArray) {
    // Require 3D hiddens [B,L,D] and 2D masks [B,L]; otherwise return the other side (avoids concatenate/dim(2) on collapsed or wrong-rank arrays).
    if hidden1.ndim < 3 || mask1.ndim < 2 { return (hidden2, mask2) }
    if hidden2.ndim < 3 || mask2.ndim < 2 { return (hidden1, mask1) }
    let L1 = hidden1.dim(1)
    let L2 = hidden2.dim(1)
    if L1 == 0 { return (hidden2, mask2) }
    if L2 == 0 { return (hidden1, mask1) }
    let hiddenCat = concatenated([hidden1, hidden2], axis: 1)
    let maskCat = concatenated([mask1, mask2], axis: 1)
    let B = hiddenCat.dim(0)
    let L = hiddenCat.dim(1)
    let D = hiddenCat.dim(2)
    hiddenCat.eval()
    maskCat.eval()
    let maskF = maskCat.asType(.float32)
    var lengths = [Int](repeating: 0, count: B)
    var sortIdx = [[Int]](repeating: [], count: B)
    for b in 0..<B {
        var row: [(Float, Int)] = []
        for i in 0..<L {
            let s = maskF[b..<(b + 1), i..<(i + 1)].squeezed(axis: 0).squeezed(axis: 0)
            let v = s.item(Float.self)
            row.append((v, i))
        }
        row.sort { a, b in a.0 > b.0 }
        sortIdx[b] = row.map(\.1)
        lengths[b] = row.filter { $0.0 > 0.5 }.count
    }
    var outF = [[Float]](repeating: [Float](repeating: 0, count: D), count: B * L)
    for b in 0..<B {
        for i in 0..<L {
            let src = sortIdx[b][i]
            let row = hiddenCat[b..<(b + 1), src..<(src + 1), 0..<D].squeezed(axis: 0).squeezed(axis: 0)
            row.eval()
            for c in 0..<D {
                outF[b * L + i][c] = row[c..<(c + 1)].squeezed(axis: 0).item(Float.self)
            }
        }
    }
    let newMask: [Float] = (0..<(B * L)).map { idx in
        let b = idx / L
        let i = idx % L
        return i < lengths[b] ? 1 : 0
    }
    let outArr = MLXArray(outF.flatMap { $0 }, [B, L, D])
    let maskArr = MLXArray(newMask, [B, L])
    return (outArr, maskArr)
}
