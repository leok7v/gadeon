import SwiftUI
import UIKit

struct RenameField: UIViewRepresentable {

    @Binding var text: String
    let onSubmit: () -> Void
    // Symbol parity with the macOS twin, which takes Esc. A phone keyboard
    // has no such key, so Cancel in the dialog is the only way out.
    let onCancel: () -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> SelectingTextField {
        let field = SelectingTextField()
        field.delegate = context.coordinator
        field.text = text
        field.font = UIFont.preferredFont(forTextStyle: .body)
        field.returnKeyType = .done
        field.clearButtonMode = .whileEditing
        field.autocorrectionType = .no
        field.addTarget(context.coordinator,
                        action: #selector(Coordinator.changed(_:)),
                        for: .editingChanged)
        return field
    }

    func updateUIView(_ field: SelectingTextField, context: Context) {
        context.coordinator.parent = self
        if field.text != text { field.text = text }
    }

    @MainActor final class Coordinator: NSObject, UITextFieldDelegate {

        var parent: RenameField

        init(_ parent: RenameField) { self.parent = parent }

        @objc func changed(_ field: UITextField) {
            parent.text = field.text ?? ""
        }

        func textFieldShouldReturn(_ field: UITextField) -> Bool {
            parent.onSubmit()
            return false
        }
    }
}

// Focus follows the field INTO the window. updateUIView cannot carry it:
// it runs before the view has a window, and a representable whose inputs
// never change again is never updated a second time, so a claim deferred
// to it is a claim that never happens.

final class SelectingTextField: UITextField {

    private var claimed = false

    override func didMoveToWindow() {
        super.didMoveToWindow()
        if window != nil, !claimed {
            claimed = true
            // One hop: selectAll needs the field to be first responder, and
            // that is not true until becomeFirstResponder has settled.
            Task { @MainActor in
                self.becomeFirstResponder()
                self.selectAll(nil)
            }
        }
    }
}
