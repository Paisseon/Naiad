//
//  NSTextField.swift
//  Naiad
//
//  Created by Lilliana on 25/12/2022.
//

#if os(macOS)
import AppKit

extension NSTextField {
    open override var focusRingType: NSFocusRingType {
        get { .none }
        set { }
    }
}
#endif
