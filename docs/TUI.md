# Terminal conversation

The full-screen terminal client renders a structured presentation of canonical conversation and task state. User and assistant messages use distinct headers and text rails; system, tool and error activity uses compact glyph-prefixed rows. This layout remains readable without color.

Ordinary and Temporary-mode assistant content streams into its Conversation entry in buffered chunks. Temporary turns remain in memory only and are discarded on exit or interruption. Raw backend reasoning, when enabled, remains a separate visibility-controlled stream. Thinking policy and reasoning visibility remain independent.

The Conversation viewer supports mouse-wheel scrolling, PageUp/PageDown and Home/End navigation. Streaming follows while the viewport is at the bottom. Scrolling upward suspends follow and marks newer output below; End returns to the bottom and resumes follow.

Conversation text is selectable. Ctrl-C copies only a bounded selection owned by the transcript and emits a write-only OSC 52 terminal clipboard operation; no clipboard-reading permission is required. Appended streaming content preserves an existing selection and scrolled viewport.

The user header defaults to the operating-system account name. `ui.display_name` can override it in configuration.
