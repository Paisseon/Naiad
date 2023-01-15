//
//  ContentView.swift
//  Naiad
//
//  Created by Lilliana on 21/12/2022.
//

import SwiftUI

#if os(iOS)
import UIKit
#endif

struct ContentView: View {
    @StateObject private var naiad: Naiad = .shared
    var body: some View {
        ZStack {
            #if os(iOS)
            if UIDevice.current.orientation.isLandscape {
                ContentView_macOS()
            } else {
                ContentView_iOS()
            }
            #else
            ContentView_macOS()
            #endif
            
            if !naiad.doesModelExist {
                DownloadView()
            }
        }
    }
}
