"use client";

/**
 * Optional-depth disclosure for how-it-works — HeroUI, Version C chrome.
 */

import { Button, Disclosure } from "@heroui/react";

export default function MethodDisclosure() {
  return (
    <Disclosure className="method-disclosure">
      <Disclosure.Heading>
        <Button slot="trigger" variant="secondary" className="method-trigger">
          Optional depth
          <Disclosure.Indicator />
        </Button>
      </Disclosure.Heading>
      <Disclosure.Content>
        <Disclosure.Body className="method-body">
          <p>
            De-vig means removing the bookmaker margin from implied probabilities
            so home / draw / away sum to 100% and can sit next to the model.
          </p>
          <p>
            Brier score measures how close probability forecasts were to the
            outcomes (lower is better). Log loss is a related proper score that
            penalises confident wrong calls more sharply.
          </p>
        </Disclosure.Body>
      </Disclosure.Content>
      <style>{`
        .method-disclosure {
          margin-top: 12px;
          max-width: 38em;
        }
        .method-trigger {
          width: 100%;
          justify-content: space-between;
          border-radius: var(--radius) !important;
          border: 1px solid var(--line) !important;
          background: var(--paper-2) !important;
          color: var(--ink) !important;
          font-size: 0.78rem !important;
          font-weight: 600 !important;
          letter-spacing: 0.04em;
          text-transform: uppercase;
          min-height: 44px;
        }
        .method-body {
          margin-top: 8px;
          padding: 12px;
          border: 1px solid var(--line-soft);
          border-radius: var(--radius);
          background: var(--paper);
          font-size: 0.88rem;
          color: var(--ink-soft);
        }
        .method-body p {
          margin: 0 0 8px;
        }
        .method-body p:last-child {
          margin-bottom: 0;
        }
      `}</style>
    </Disclosure>
  );
}
