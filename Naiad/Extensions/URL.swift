//
//  URL.swift
//  Naiad
//
//  Created by Lilliana on 22/12/2022.
//

import Foundation

extension URL {
    func slash(
        _ nextPath: String
    ) -> URL {
        self.appendingPathComponent(nextPath)
    }
}
