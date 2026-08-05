import type { AgentConnectionQueueState } from "../agent-connection/types.js";

export type PendingMessageLane = "steering" | "followUp";
export interface PendingMessageLocation {
	id: string;
	lane: PendingMessageLane;
	index: number;
}
export interface PendingMessageChange {
	draft: string;
	selected?: PendingMessageLocation;
}

type Item = PendingMessageLocation & { id: string; text: string };
const clone = (queue: AgentConnectionQueueState): AgentConnectionQueueState => ({
	steering: [...queue.steering],
	followUp: [...queue.followUp],
	...(queue.revision !== undefined ? { revision: queue.revision } : {}),
	...(queue.items ? { items: queue.items.map((item) => ({ ...item })) } : {}),
});
const items = (queue: AgentConnectionQueueState): Item[] =>
	queue.items
		? [...queue.items].sort((a, b) => (a.lane === b.lane ? a.index - b.index : a.lane === "steering" ? -1 : 1))
		: [
				...queue.steering.map((text, index) => ({
					id: `legacy:steering:${index}`,
					lane: "steering" as const,
					index,
					text,
				})),
				...queue.followUp.map((text, index) => ({
					id: `legacy:followUp:${index}`,
					lane: "followUp" as const,
					index,
					text,
				})),
			];
const equal = (a: AgentConnectionQueueState, b: AgentConnectionQueueState): boolean => {
	const left = items(a);
	const right = items(b);
	return (
		a.steering.length === b.steering.length &&
		a.followUp.length === b.followUp.length &&
		a.steering.every((text, index) => text === b.steering[index]) &&
		a.followUp.every((text, index) => text === b.followUp[index]) &&
		left.length === right.length &&
		left.every(
			(item, index) =>
				item.id === right[index]?.id &&
				item.lane === right[index]?.lane &&
				item.index === right[index]?.index &&
				item.text === right[index]?.text,
		)
	);
};

/** Pending-message history plus all pure mutations of its selected item. */
export class PendingMessageNavigation {
	private queue?: AgentConnectionQueueState;
	private cursor = 0;
	private draft = "";
	private edits = new Map<string, string>();

	get selected(): PendingMessageLocation | undefined {
		const item = this.queue && items(this.queue)[this.cursor];
		return item ? { id: item.id, lane: item.lane, index: item.index } : undefined;
	}
	get isAtDraft(): boolean {
		return !!this.queue && this.cursor === items(this.queue).length;
	}

	reconcile(queue: AgentConnectionQueueState, selectedId?: string): void {
		if (!selectedId) {
			this.reset();
			return;
		}
		const cursor = items(queue).findIndex((item) => item.id === selectedId);
		if (cursor < 0) {
			this.reset();
			return;
		}
		this.queue = clone(queue);
		this.cursor = cursor;
	}

	checkpoint(): { queue?: AgentConnectionQueueState; cursor: number; draft: string; edits: Map<string, string> } {
		return {
			queue: this.queue && clone(this.queue),
			cursor: this.cursor,
			draft: this.draft,
			edits: new Map(this.edits),
		};
	}

	restore(state: ReturnType<PendingMessageNavigation["checkpoint"]>): void {
		this.queue = state.queue;
		this.cursor = state.cursor;
		this.draft = state.draft;
		this.edits = state.edits;
	}

	reset(): void {
		this.queue = undefined;
		this.cursor = 0;
		this.draft = "";
		this.edits.clear();
	}

	sync(queue: AgentConnectionQueueState): string | undefined {
		if (!this.queue) return undefined;
		const previous = this.queue;
		const selectedId = this.selected?.id;
		const atDraft = this.cursor === items(previous).length;
		if (equal(previous, queue)) {
			// The content may be unchanged while a newer CAS revision arrived.
			this.queue = clone(queue);
			if (atDraft) this.cursor = items(this.queue).length;
			return undefined;
		}
		if (atDraft) {
			this.queue = clone(queue);
			this.cursor = items(this.queue).length;
			return undefined;
		}
		if (selectedId) {
			const cursor = items(queue).findIndex((item) => item.id === selectedId);
			if (cursor >= 0) {
				this.queue = clone(queue);
				this.cursor = cursor;
				return undefined;
			}
		}
		const draft = this.draft;
		this.reset();
		return draft;
	}

	browse(queue: AgentConnectionQueueState, text: string, delta: -1 | 1): string | undefined {
		if (!this.queue || !equal(this.queue, queue)) {
			if (delta > 0 || items(queue).length === 0) return undefined;
			this.queue = clone(queue);
			this.cursor = items(queue).length;
			this.draft = text;
			this.edits.clear();
		} else this.capture(text);
		const end = items(this.queue).length;
		this.cursor = Math.max(0, Math.min(end, this.cursor + delta));
		return this.cursor === end ? this.draft : this.value(this.cursor);
	}

	private value(index: number): string {
		const item = items(this.queue!)[index]!;
		return this.edits.get(item.id) ?? item.text;
	}

	capture(text: string): void {
		if (!this.queue) return;
		if (this.cursor < items(this.queue).length) this.edits.set(items(this.queue)[this.cursor]!.id, text);
		else this.draft = text;
	}

	/** Mutate the selected item, preserving its edit on reorder and returning to the draft otherwise. */
	change(kind: "delete" | "followUp" | "steer" | "earlier" | "later", text: string): PendingMessageChange | undefined {
		const selected = this.selected;
		if (!this.queue || !selected) return undefined;
		const queue = clone(this.queue);
		if (kind === "earlier" || kind === "later") {
			const target = selected.index + (kind === "earlier" ? -1 : 1);
			const lane = queue[selected.lane];
			const editableCount = queue.items?.filter((item) => item.lane === selected.lane).length ?? lane.length;
			if (target < 0 || target >= editableCount) return undefined;
			if (!queue.items) [lane[selected.index], lane[target]] = [lane[target]!, lane[selected.index]!];
			if (queue.items) {
				const other = queue.items.find((item) => item.lane === selected.lane && item.index === target);
				const current = queue.items.find((item) => item.id === selected.id);
				if (other) other.index = selected.index;
				if (current) current.index = target;
			}
			const moved = { id: selected.id, lane: selected.lane, index: target };
			this.queue = queue;
			this.cursor = items(queue).findIndex((item) => item.id === selected.id);
			this.edits.set(selected.id, text);
			return { draft: this.draft, selected: moved };
		}
		const draft = this.draft;
		this.reset();
		return { draft };
	}
}
