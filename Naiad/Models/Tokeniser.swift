//
//  Tokeniser.swift
//  Naiad
//
//  Created by Lilliana on 22/12/2022.
//

import Foundation

final class Tokeniser {
    // MARK: Lifecycle

    init(
        at model: URL
    ) {
        var vocabList: [String] = []

        for i in Array(33 ... 126) + Array(161 ... 172) + Array(174 ... 255) {
            uniBytes[i] = Character(Unicode.Scalar(i)!)
            vocabList.append(String(Unicode.Scalar(i)!))
        }

        for i in 0 ... 255 {
            if uniBytes[i] != nil {
                continue
            }

            uniBytes[i] = Character(Unicode.Scalar(256 + uniBytes.count - 188)!)
            vocabList.append(String(uniBytes[i]!))
        }

        vocabList += vocabList.map { $0 + "</w>" }

        guard let vocabFile: String = try? .init(contentsOf: model.slash("bpe_simple_vocab_16e6.txt")) else {
            return
        }

        for (i, m) in vocabFile.split(separator: "\n")[1 ..< 0xBEFF].enumerated() {
            ranks[String(m)] = i
            vocabList.append(m.split(separator: " ").joined(separator: ""))
        }

        vocab = vocabList.enumerated().reduce(into: [:]) {
            $0[$1.element] = $1.offset
        }
    }

    // MARK: Internal

    func encode(
        token: String
    ) -> [Int] {
        let str: String = NSString(string: clean(token.lowercased())) as String
        var bpe: [Int] = []

        pattern.matches(in: str, range: NSRange(location: 0, length: str.count)).forEach { match in
            if let range: Range<String.Index> = .init(match.range, in: str) {
                bpe.append(contentsOf: _encode(String(str[range])))
            }
        }

        return [0xC0FE] + bpe[..<min(75, bpe.count)] + [Int](repeating: 0xC0FF, count: max(1, 76 - bpe.count))
    }

    // MARK: Private

    private let pattern: NSRegularExpression = try! .init(pattern: #"'s|'t|'re|'ve|'m|'ll|'d|[^\s]+"#, options: .caseInsensitive)
    private var ranks: [String: Int] = [:]
    private var uniBytes: [Int: Character] = [:]
    private var vocab: [String: Int] = [:]

    private func clean(
        _ s: String
    ) -> String {
        s.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func getPairs(
        from s: [String]
    ) -> Set<String> {
        Set<String>((1 ..< s.count).map { s[$0 - 1] + " " + s[$0] })
    }

    private func _encode(
        _ _token: String
    ) -> [Int] {
        let token: String = .init(_token.utf8.map { uniBytes[Int($0)]! })
        var word: [String] = token[..<token.index(before: token.endIndex)].map { String($0) } + [token.suffix(from: token.index(before: token.endIndex)) + "</w>"]
        var pairs: Set<String> = getPairs(from: Array(word))
        var merged: [String] = [token + "</w>"]
        var count: Int = 0

        if !pairs.isEmpty {
            while true {
                count += 1

                guard count < 0x2000,
                      let highBigram: String = pairs.min(by: { ranks[$0, default: .max] < ranks[$1, default: .max] }),
                      ranks[highBigram] != nil
                else {
                    break
                }

                let fs: [String.SubSequence] = highBigram.split(separator: " ")
                let (first, second): (String, String) = (String(fs[0]), String(fs[1]))
                var (newWord, i): ([String], Int) = ([], 0)

                while i < word.count {
                    guard let j: ArraySlice<String>.Index = word[i ..< word.count].firstIndex(of: first) else {
                        newWord.append(contentsOf: word[i ..< word.count])
                        break
                    }

                    newWord.append(contentsOf: word[i ..< j])
                    i = j

                    if word[i] == first, word[i + 1] == second {
                        newWord.append(first + second)
                        i += 2
                    } else {
                        newWord.append(word[i])
                        i += 1
                    }
                }

                word = newWord

                if word.count == 1 {
                    break
                } else {
                    pairs = getPairs(from: word)
                }
            }

            merged = word
        }

        return merged.map { vocab[$0]! }
    }
}
