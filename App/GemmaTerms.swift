import LLM
import SwiftUI

// Google's Gemma Terms of Use, shown BEFORE gemma-4 is downloaded.
//
// WHY this exists when no other model has one: every other entry in the
// catalog is Apache-2.0, which asks nothing of an end user. Gemma is not --
// its terms bind whoever uses the model, and they oblige a distributor to
// pass that on. Bundling it silently would leave a user bound by terms they
// were never shown, so consent is a gate rather than a footnote.
//
// Deliberately NOT a reproduction of the licence: linking the canonical text
// is both what Google asks and the only version that cannot go stale here.
enum GemmaTerms {
    // The catalog names it; matching on the name keeps the gate with the
    // model rather than scattered through the download flow.
    static let modelName = "gemma-4-E2B"

    static func applies(to name: String) -> Bool { name == modelName }

    static let termsURL = URL(string: "https://ai.google.dev/gemma/terms")!
    static let policyURL =
        URL(string: "https://ai.google.dev/gemma/prohibited_use_policy")!

    // Persisted per model, so the gate is shown once and a later re-download
    // (or a delete and fetch again) does not re-ask.
    private static let key = "gemmaTermsAccepted"

    static var accepted: Bool {
        UserDefaults.standard.bool(forKey: key)
    }

    static func accept() {
        UserDefaults.standard.set(true, forKey: key)
    }
}

// Shown in place of the download-consent panel when the chosen model carries
// the Gemma terms. Agreeing leads INTO the normal size/consent panel rather
// than starting a download itself: the licence and the "this costs you 2.7 GB"
// question are different consents and should not be collapsed into one tap.
struct GemmaTermsView: View {
    let onAgree: () -> Void
    let onCancel: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "doc.text")
                .font(.largeTitle)
                .foregroundStyle(.secondary)
            Text("Gemma Terms of Use").font(.title2).bold()
            Text("\(Models.display(GemmaTerms.modelName)) is provided by "
               + "Google under the Gemma Terms of Use, not the open-source "
               + "licence the other models use. By continuing you agree to "
               + "those terms and to the Prohibited Use Policy, which "
               + "restrict what the model may be used for.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            VStack(spacing: 6) {
                Link("Gemma Terms of Use", destination: GemmaTerms.termsURL)
                Link("Prohibited Use Policy", destination: GemmaTerms.policyURL)
            }
            .font(.callout)
            Text("Gemma is a Google model. This app is not affiliated with "
               + "or endorsed by Google.")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
            VStack(spacing: 14) {
                Button("Agree and Continue", action: onAgree)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .keyboardShortcut(.defaultAction)
                Button("Cancel", action: onCancel)
                    .controlSize(.large)
                    .keyboardShortcut(.cancelAction)
            }
            .padding(.top, 8)
        }
        .padding(32)
        .frame(maxWidth: 460)
    }
}
