/**
 * Tests for AgentSession concurrent prompt guard.
 */

import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent } from "@earendil-works/pi-agent-core";
import {
	type AssistantMessage,
	type AssistantMessageEvent,
	EventStream,
	getModel,
	type ImageContent,
	type TextContent,
} from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AgentSession } from "../src/core/agent-session.js";
import { AuthStorage } from "../src/core/auth-storage.js";
import { ModelRegistry } from "../src/core/model-registry.js";
import { SessionManager } from "../src/core/session-manager.js";
import { SettingsManager } from "../src/core/settings-manager.js";
import type { BuildSystemPromptOptions } from "../src/core/system-prompt.js";
import { createTestExtensionsResult, createTestResourceLoader } from "./utilities.js";

// Mock stream that mimics AssistantMessageEventStream
class MockAssistantStream extends EventStream<AssistantMessageEvent, AssistantMessage> {
	constructor() {
		super(
			(event) => event.type === "done" || event.type === "error",
			(event) => {
				if (event.type === "done") return event.message;
				if (event.type === "error") return event.error;
				throw new Error("Unexpected event type");
			},
		);
	}
}

function createAssistantMessage(text: string): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "mock",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

describe("AgentSession concurrent prompt guard", () => {
	let session: AgentSession;
	let tempDir: string;

	beforeEach(() => {
		tempDir = join(tmpdir(), `pi-concurrent-test-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });
	});

	afterEach(async () => {
		delete (globalThis as typeof globalThis & { testExtensionApi?: unknown }).testExtensionApi;
		delete (globalThis as typeof globalThis & { testCommandRuns?: unknown }).testCommandRuns;
		if (session) {
			session.dispose();
		}
		if (tempDir && existsSync(tempDir)) {
			rmSync(tempDir, { recursive: true });
		}
	});

	function createSession() {
		const model = getModel("anthropic", "claude-sonnet-4-5")!;
		let abortSignal: AbortSignal | undefined;

		// Use a stream function that responds to abort
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model,
				systemPrompt: "Test",
				tools: [],
			},
			streamFn: (_model, _context, options) => {
				abortSignal = options?.signal;
				const stream = new MockAssistantStream();
				queueMicrotask(() => {
					stream.push({ type: "start", partial: createAssistantMessage("") });
					const checkAbort = () => {
						if (abortSignal?.aborted) {
							stream.push({ type: "error", reason: "aborted", error: createAssistantMessage("Aborted") });
						} else {
							setTimeout(checkAbort, 5);
						}
					};
					checkAbort();
				});
				return stream;
			},
		});

		const sessionManager = SessionManager.inMemory();
		const settingsManager = SettingsManager.create(tempDir, tempDir);
		const authStorage = AuthStorage.create(join(tempDir, "auth.json"));
		const modelRegistry = ModelRegistry.create(authStorage, tempDir);
		// Set a runtime API key so validation passes
		authStorage.setRuntimeApiKey("anthropic", "test-key");

		session = new AgentSession({
			agent,
			sessionManager,
			settingsManager,
			cwd: tempDir,
			modelRegistry,
			resourceLoader: createTestResourceLoader(),
		});

		return session;
	}

	it("should throw when prompt() called while streaming", async () => {
		createSession();

		// Start first prompt (don't await, it will block until abort)
		const firstPrompt = session.prompt("First message");

		// Wait a tick for isStreaming to be set
		await new Promise((resolve) => setTimeout(resolve, 10));

		// Verify we're streaming
		expect(session.isStreaming).toBe(true);

		// Second prompt should reject
		await expect(session.prompt("Second message")).rejects.toThrow(
			"Agent is already processing. Specify streamingBehavior ('steer' or 'followUp') to queue the message.",
		);

		// Cleanup
		await session.abort();
		await firstPrompt.catch(() => {}); // Ignore abort error
	});

	it("does not bump queue revision for active phase-only updates", async () => {
		createSession();
		const revisions: number[] = [];
		const unsubscribe = session.subscribe((event) => {
			if (event.type === "session_action_update" && event.actions.active)
				revisions.push(event.actions.revision ?? -1);
		});
		const prompt = session.prompt("running");
		await new Promise((resolve) => setTimeout(resolve, 10));
		expect(new Set(revisions).size).toBeLessThanOrEqual(1);
		unsubscribe();
		await session.abort();
		await prompt.catch(() => {});
	});

	it("should allow steer() while streaming", async () => {
		createSession();

		// Start first prompt
		const firstPrompt = session.prompt("First message");
		await new Promise((resolve) => setTimeout(resolve, 10));

		// steer should work while streaming
		expect(() => session.steer("Steering message")).not.toThrow();
		expect(session.queuedActionCount).toBe(1);

		// Cleanup
		await session.abort();
		await firstPrompt.catch(() => {});
	});

	it("should allow followUp() while streaming", async () => {
		createSession();

		// Start first prompt
		const firstPrompt = session.prompt("First message");
		await new Promise((resolve) => setTimeout(resolve, 10));

		// followUp should work while streaming
		expect(() => session.followUp("Follow-up message")).not.toThrow();
		expect(session.queuedActionCount).toBe(1);

		// Cleanup
		await session.abort();
		await firstPrompt.catch(() => {});
	});

	it("atomically mutates editable queued actions without starting a new turn", async () => {
		createSession();
		const firstPrompt = session.prompt("running");
		await new Promise((resolve) => setTimeout(resolve, 10));
		await session.prompt("duplicate", { streamingBehavior: "steer", queueIfBusy: true, source: "interactive" });
		await session.sendUserMessage("extension item", { deliverAs: "steer" });
		await session.prompt("/compact focus on tools", { streamingBehavior: "steer", queueIfBusy: true });
		const originalImage: ImageContent = { type: "image", data: "original", mimeType: "image/png" };
		await session.prompt("duplicate [image #1]", {
			streamingBehavior: "steer",
			queueIfBusy: true,
			source: "interactive",
			images: [originalImage],
		});
		await session.prompt("follow", { streamingBehavior: "followUp", queueIfBusy: true, source: "interactive" });
		const staleRevision = session.queueRevision - 1;
		const [first, extension, command, second, follow] = session.getEditableQueueItems();
		expect([first?.text, extension?.text, command?.text, second?.text, follow?.text]).toEqual([
			"duplicate",
			"extension item",
			"/compact focus on tools",
			"duplicate [image #1]",
			"follow",
		]);

		expect(session.getSteeringMessages()).toEqual([
			"duplicate",
			"extension item",
			"/compact focus on tools",
			"duplicate [image #1]",
		]);
		const boundaryRevision = session.queueRevision;
		expect(session.mutateQueuedUserMessage(first!.id, boundaryRevision, { type: "move_earlier" })).toBe("noop");
		expect(session.queueRevision).toBe(boundaryRevision);
		expect(session.mutateQueuedUserMessage(second!.id, staleRevision, { type: "delete" })).toBe("stale");
		expect(session.mutateQueuedUserMessage("missing", session.queueRevision, { type: "delete" })).toBe("stale");
		expect(session.mutateQueuedUserMessage(second!.id, session.queueRevision, { type: "move_earlier" })).toBe(
			"applied",
		);
		expect(session.getEditableQueueItems().map((item) => item.id)).toEqual([
			first!.id,
			extension!.id,
			second!.id,
			command!.id,
			follow!.id,
		]);
		expect(
			session.mutateQueuedUserMessage(command!.id, session.queueRevision, {
				type: "replace_follow_up",
				text: "/compact revised focus",
			}),
		).toBe("applied");
		expect(session.getEditableQueueItems().find((item) => item.id === command!.id)?.lane).toBe("followUp");
		const editedCommand = session
			.getSessionActionRecoverySnapshot()
			.actions.find((action) => action.id === command!.id);
		expect(editedCommand?.payload).toMatchObject({
			kind: "session_command",
			text: "/compact revised focus",
			command: { name: "compact", args: "revised focus" },
		});
		const commandRevision = session.queueRevision;
		expect(
			session.mutateQueuedUserMessage(command!.id, commandRevision, {
				type: "replace_steering",
				text: "not a session command",
			}),
		).toBe("invalid");
		expect(session.queueRevision).toBe(commandRevision);
		const commandImage: ImageContent = { type: "image", data: "command", mimeType: "image/png" };
		expect(
			session.mutateQueuedUserMessage(command!.id, session.queueRevision, {
				type: "replace_steering",
				text: "/compact revised focus",
				images: [commandImage],
			}),
		).toBe("applied");
		expect(
			session.mutateQueuedUserMessage(command!.id, session.queueRevision, {
				type: "replace_steering",
				text: "/compact preserve image",
			}),
		).toBe("applied");
		let commandPayload = session
			.getSessionActionRecoverySnapshot()
			.actions.find((action) => action.id === command!.id)?.payload;
		expect(commandPayload).toMatchObject({ images: [commandImage] });
		if (commandPayload?.kind === "session_command") expect(commandPayload.images?.[0]).not.toBe(commandImage);
		expect(
			session.mutateQueuedUserMessage(command!.id, session.queueRevision, {
				type: "replace_steering",
				text: "/compact clear image",
				images: [],
			}),
		).toBe("applied");
		commandPayload = session
			.getSessionActionRecoverySnapshot()
			.actions.find((action) => action.id === command!.id)?.payload;
		expect(commandPayload).not.toHaveProperty("images");
		expect(session.mutateQueuedUserMessage(command!.id, session.queueRevision, { type: "move_earlier" })).toBe(
			"applied",
		);
		expect(session.mutateQueuedUserMessage(command!.id, session.queueRevision, { type: "move_later" })).toBe(
			"applied",
		);
		expect(
			session.mutateQueuedUserMessage(second!.id, session.queueRevision, {
				type: "replace_steering",
				text: "reconnected edit [image #1]",
			}),
		).toBe("applied");
		let preserved = session.getSessionActionRecoverySnapshot().actions.find((action) => action.id === second!.id);
		expect(preserved?.payload).toMatchObject({ images: [originalImage] });
		const imageOnlyRevision = session.queueRevision;
		const imageOnlyReplacement: ImageContent = { type: "image", data: "replacement", mimeType: "image/png" };
		expect(
			session.mutateQueuedUserMessage(second!.id, imageOnlyRevision, {
				type: "replace_steering",
				text: "reconnected edit [image #1]",
				images: [imageOnlyReplacement],
			}),
		).toBe("applied");
		expect(session.queueRevision).toBe(imageOnlyRevision + 1);
		expect(
			session.mutateQueuedUserMessage(second!.id, session.queueRevision, {
				type: "replace_steering",
				text: "removed marker",
				images: [],
			}),
		).toBe("applied");
		preserved = session.getSessionActionRecoverySnapshot().actions.find((action) => action.id === second!.id);
		expect(preserved?.payload).not.toHaveProperty("images");
		const image: ImageContent = { type: "image", data: "encoded", mimeType: "image/png" };
		expect(
			session.mutateQueuedUserMessage(second!.id, session.queueRevision, {
				type: "replace_follow_up",
				text: "converted",
				images: [image],
			}),
		).toBe("applied");
		expect(session.getEditableQueueItems()).toMatchObject([
			{ id: first!.id, lane: "steering", text: "duplicate" },
			{ id: extension!.id, lane: "steering", text: "extension item" },
			{ id: command!.id, lane: "steering", text: "/compact clear image" },
			{ id: follow!.id, lane: "followUp", text: "follow" },
			{ id: second!.id, lane: "followUp", text: "converted" },
		]);
		const recovered = session.getSessionActionRecoverySnapshot().actions.find((action) => action.id === second!.id);
		expect(recovered?.payload).toMatchObject({ text: "converted", images: [image] });
		expect(recovered?.payload.kind === "turn" ? recovered.payload.records[0]?.message.content : undefined).toEqual([
			{ type: "text", text: "converted" },
			image,
		]);
		expect(session.mutateQueuedUserMessage(command!.id, session.queueRevision, { type: "delete" })).toBe("applied");
		expect(session.mutateQueuedUserMessage(second!.id, session.queueRevision, { type: "delete" })).toBe("applied");
		expect(session.mutateQueuedUserMessage(second!.id, session.queueRevision, { type: "delete" })).toBe("stale");
		expect(session.isStreaming).toBe(true);

		await session.abort();
		await firstPrompt.catch(() => {});
	});

	it("should queue extension-origin steering messages while streaming", async () => {
		const model = getModel("anthropic", "claude-sonnet-4-5")!;
		let abortSignal: AbortSignal | undefined;
		let sawSteeringMessage = false;
		let lastInputSource: string | undefined;
		const queueEvents: Array<{ steering: readonly string[]; followUp: readonly string[] }> = [];

		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model,
				systemPrompt: "Test",
				tools: [],
			},
			streamFn: (_model, context, options) => {
				abortSignal = options?.signal;
				const stream = new MockAssistantStream();
				queueMicrotask(() => {
					const userTexts = context.messages
						.filter((message) => message.role === "user")
						.map((message) => {
							if (typeof message.content === "string") {
								return message.content;
							}
							return message.content
								.filter((part): part is TextContent | ImageContent => typeof part === "object" && part !== null)
								.filter((part): part is TextContent => part.type === "text")
								.map((part) => part.text)
								.join("\n");
						});

					if (userTexts.includes("Steer from extension")) {
						sawSteeringMessage = true;
						stream.push({ type: "start", partial: createAssistantMessage("") });
						stream.push({ type: "done", reason: "stop", message: createAssistantMessage("Steered") });
						return;
					}

					stream.push({ type: "start", partial: createAssistantMessage("") });
					const checkAbort = () => {
						if (abortSignal?.aborted) {
							stream.push({ type: "error", reason: "aborted", error: createAssistantMessage("Aborted") });
						} else {
							setTimeout(checkAbort, 5);
						}
					};
					checkAbort();
				});
				return stream;
			},
		});

		const sessionManager = SessionManager.inMemory();
		const settingsManager = SettingsManager.create(tempDir, tempDir);
		const authStorage = AuthStorage.create(join(tempDir, "auth.json"));
		const modelRegistry = ModelRegistry.create(authStorage, tempDir);
		authStorage.setRuntimeApiKey("anthropic", "test-key");

		const extensionsResult = await createTestExtensionsResult([
			(pi) => {
				(globalThis as typeof globalThis & { testExtensionApi?: unknown }).testExtensionApi = pi;
			},
			(pi) => {
				pi.on("input", async (event) => {
					lastInputSource = event.source;
				});
			},
		]);

		session = new AgentSession({
			agent,
			sessionManager,
			settingsManager,
			cwd: tempDir,
			modelRegistry,
			resourceLoader: createTestResourceLoader({ extensionsResult }),
		});
		session.subscribe((event) => {
			if (event.type === "session_action_update") {
				queueEvents.push({ steering: event.actions.steering, followUp: event.actions.followUps });
			}
		});

		const firstPrompt = session.prompt("First message");
		await new Promise((resolve) => setTimeout(resolve, 10));
		expect(session.isStreaming).toBe(true);

		const pi = (
			globalThis as typeof globalThis & {
				testExtensionApi?: {
					sendUserMessage: (content: string, options?: { deliverAs?: "steer" | "followUp" }) => void;
				};
			}
		).testExtensionApi;
		expect(pi).toBeDefined();

		pi!.sendUserMessage("Steer from extension", { deliverAs: "steer" });
		await new Promise((resolve) => setTimeout(resolve, 25));

		expect(session.queuedActionCount).toBe(1);
		expect(session.getSteeringMessages()).toContain("Steer from extension");
		expect(lastInputSource).toBe("extension");
		expect(queueEvents.some((event) => event.steering.includes("Steer from extension"))).toBe(true);

		await session.abort();
		await firstPrompt.catch(() => {});

		expect(sawSteeringMessage).toBe(false);
		expect(session.getSteeringMessages()).toContain("Steer from extension");
		expect(session.queuedActionCount).toBe(1);

		await session.prompt("After abort");

		expect(sawSteeringMessage).toBe(true);
		expect(session.queuedActionCount).toBe(0);
	});

	it("delivers accepted agent messages without extension input interception", async () => {
		const model = getModel("anthropic", "claude-sonnet-4-5")!;
		let inputCalls = 0;
		let receivedUserText: string | undefined;
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model,
				systemPrompt: "Test",
				tools: [],
			},
			streamFn: (_model, context) => {
				const stream = new MockAssistantStream();
				queueMicrotask(() => {
					const userMessages = context.messages.filter((message) => message.role === "user");
					const user = userMessages.at(-1);
					if (user && typeof user.content !== "string") {
						receivedUserText = user.content
							.filter(
								(part): part is TextContent =>
									typeof part === "object" && part !== null && part.type === "text",
							)
							.map((part) => part.text)
							.join("\n");
					}
					stream.push({ type: "start", partial: createAssistantMessage("") });
					stream.push({ type: "done", reason: "stop", message: createAssistantMessage("Delivered") });
				});
				return stream;
			},
		});

		const sessionManager = SessionManager.inMemory();
		const settingsManager = SettingsManager.create(tempDir, tempDir);
		const authStorage = AuthStorage.create(join(tempDir, "auth.json"));
		const modelRegistry = ModelRegistry.create(authStorage, tempDir);
		authStorage.setRuntimeApiKey("anthropic", "test-key");
		const extensionsResult = await createTestExtensionsResult([
			(pi) => {
				pi.on("input", async () => {
					inputCalls++;
					return { action: "handled" };
				});
			},
		]);

		session = new AgentSession({
			agent,
			sessionManager,
			settingsManager,
			cwd: tempDir,
			modelRegistry,
			resourceLoader: createTestResourceLoader({ extensionsResult }),
		});

		await session.acceptAgentMessagePrompt("agent-to-agent payload", { expandPromptTemplates: false });
		await session.agent.waitForIdle();

		expect(inputCalls).toBe(0);
		expect(receivedUserText).toBe("agent-to-agent payload");
	});

	it("delivers internal prompts without extension input interception", async () => {
		const model = getModel("anthropic", "claude-sonnet-4-5")!;
		let inputCalls = 0;
		let receivedUserText: string | undefined;
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model,
				systemPrompt: "Test",
				tools: [],
			},
			streamFn: (_model, context) => {
				const stream = new MockAssistantStream();
				queueMicrotask(() => {
					const userMessages = context.messages.filter((message) => message.role === "user");
					const user = userMessages.at(-1);
					if (user && typeof user.content !== "string") {
						receivedUserText = user.content
							.filter(
								(part): part is TextContent =>
									typeof part === "object" && part !== null && part.type === "text",
							)
							.map((part) => part.text)
							.join("\n");
					}
					stream.push({ type: "start", partial: createAssistantMessage("") });
					stream.push({ type: "done", reason: "stop", message: createAssistantMessage("Delivered") });
				});
				return stream;
			},
		});

		const sessionManager = SessionManager.inMemory();
		const settingsManager = SettingsManager.create(tempDir, tempDir);
		const authStorage = AuthStorage.create(join(tempDir, "auth.json"));
		const modelRegistry = ModelRegistry.create(authStorage, tempDir);
		authStorage.setRuntimeApiKey("anthropic", "test-key");
		const extensionsResult = await createTestExtensionsResult([
			(pi) => {
				pi.on("input", async () => {
					inputCalls++;
					return { action: "handled" };
				});
			},
		]);

		session = new AgentSession({
			agent,
			sessionManager,
			settingsManager,
			cwd: tempDir,
			modelRegistry,
			resourceLoader: createTestResourceLoader({ extensionsResult }),
		});

		await session.prompt("host gate follow-up", { internalPrompt: true });
		await session.agent.waitForIdle();

		expect(inputCalls).toBe(0);
		expect(receivedUserText).toBe("host gate follow-up");
	});

	it("should allow prompt() after previous completes", async () => {
		// Create session with a stream that completes immediately
		const model = getModel("anthropic", "claude-sonnet-4-5")!;
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model,
				systemPrompt: "Test",
				tools: [],
			},
			streamFn: () => {
				const stream = new MockAssistantStream();
				queueMicrotask(() => {
					stream.push({ type: "start", partial: createAssistantMessage("") });
					stream.push({ type: "done", reason: "stop", message: createAssistantMessage("Done") });
				});
				return stream;
			},
		});

		const sessionManager = SessionManager.inMemory();
		const settingsManager = SettingsManager.create(tempDir, tempDir);
		const authStorage = AuthStorage.create(join(tempDir, "auth.json"));
		const modelRegistry = ModelRegistry.create(authStorage, tempDir);
		authStorage.setRuntimeApiKey("anthropic", "test-key");

		session = new AgentSession({
			agent,
			sessionManager,
			settingsManager,
			cwd: tempDir,
			modelRegistry,
			resourceLoader: createTestResourceLoader(),
		});

		// First prompt completes
		await session.prompt("First message");

		// Should not be streaming anymore
		expect(session.isStreaming).toBe(false);

		// Second prompt should work
		await expect(session.prompt("Second message")).resolves.not.toThrow();
	});

	it("should wait for queued agent events before emitting tool_call", async () => {
		const model = getModel("anthropic", "claude-sonnet-4-5")!;
		const tool = {
			name: "dummy",
			description: "Dummy tool",
			label: "dummy",
			parameters: Type.Object({ q: Type.String() }),
			execute: async (_toolCallId: string, params: unknown) => {
				const q =
					typeof params === "object" && params !== null && "q" in params
						? String((params as { q: unknown }).q)
						: "";
				return {
					content: [{ type: "text" as const, text: `result:${q}` }],
					details: {},
				};
			},
		};

		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model,
				systemPrompt: "Test",
				tools: [tool],
			},
			streamFn: async (_model, context) => {
				const stream = new MockAssistantStream();
				queueMicrotask(() => {
					const toolResultCount = context.messages.filter((message) => message.role === "toolResult").length;
					if (toolResultCount > 0) {
						const message: AssistantMessage = {
							role: "assistant",
							content: [{ type: "text", text: "done" }],
							api: "anthropic-messages",
							provider: "anthropic",
							model: "mock",
							usage: {
								input: 1,
								output: 1,
								cacheRead: 0,
								cacheWrite: 0,
								totalTokens: 2,
								cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
							},
							stopReason: "stop",
							timestamp: Date.now(),
						};
						stream.push({ type: "start", partial: { ...message, content: [] } });
						stream.push({ type: "done", reason: "stop", message });
						return;
					}

					const message: AssistantMessage = {
						role: "assistant",
						content: [
							{ type: "toolCall", id: "toolu_1", name: "dummy", arguments: { q: "x" } },
							{ type: "toolCall", id: "toolu_2", name: "dummy", arguments: { q: "y" } },
						],
						api: "anthropic-messages",
						provider: "anthropic",
						model: "mock",
						usage: {
							input: 1,
							output: 1,
							cacheRead: 0,
							cacheWrite: 0,
							totalTokens: 2,
							cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
						},
						stopReason: "toolUse",
						timestamp: Date.now(),
					};

					stream.push({ type: "start", partial: { ...message, content: [] } });
					stream.push({ type: "done", reason: "toolUse", message });
				});
				return stream;
			},
		});

		const sessionManager = SessionManager.inMemory();
		const settingsManager = SettingsManager.create(tempDir, tempDir);
		const authStorage = AuthStorage.create(join(tempDir, "auth.json"));
		const modelRegistry = ModelRegistry.create(authStorage, tempDir);
		authStorage.setRuntimeApiKey("anthropic", "test-key");

		session = new AgentSession({
			agent,
			sessionManager,
			settingsManager,
			cwd: tempDir,
			modelRegistry,
			resourceLoader: createTestResourceLoader(),
			baseToolsOverride: { dummy: tool },
		});

		const snapshots: string[][] = [];
		const sessionWithRunner = session as unknown as {
			_extensionRunner?: {
				hasHandlers: (eventType: string) => boolean;
				emit: (event: { type: string; message?: { role?: string } }) => Promise<void>;
				emitMessageEnd: (event: { type: string; message?: { role?: string } }) => Promise<undefined>;
				emitToolCall: (event: { type: string; toolCallId: string }) => Promise<undefined>;
				emitInput: (
					text: string,
					images: unknown,
					source: "interactive" | "rpc" | "extension",
				) => Promise<{ action: "continue" }>;
				emitBeforeAgentStart: (
					prompt: string,
					images: unknown,
					systemPrompt: string,
					systemPromptOptions: BuildSystemPromptOptions,
				) => Promise<undefined>;
				invalidate: (message?: string) => void;
			};
		};
		sessionWithRunner._extensionRunner = {
			hasHandlers: (eventType) => eventType === "tool_call",
			emit: async () => {},
			emitMessageEnd: async () => undefined,
			emitToolCall: async () => {
				snapshots.push(
					sessionManager
						.getEntries()
						.filter((entry) => entry.type === "message")
						.map((entry) => entry.message.role),
				);
				return undefined;
			},
			emitInput: async () => ({ action: "continue" }),
			emitBeforeAgentStart: async () => undefined,
			invalidate: () => {},
		};

		await session.prompt("hi");
		await session.agent.waitForIdle();

		expect(snapshots).toEqual([
			["user", "assistant"],
			["user", "assistant"],
		]);
	});

	it("should persist message_end events in order with slow extension handlers", async () => {
		const model = getModel("anthropic", "claude-sonnet-4-5")!;
		const tool = {
			name: "dummy",
			description: "Dummy tool",
			label: "dummy",
			parameters: Type.Object({ q: Type.String() }),
			execute: async (_toolCallId: string, params: unknown) => {
				const q =
					typeof params === "object" && params !== null && "q" in params
						? String((params as { q: unknown }).q)
						: "";
				return {
					content: [{ type: "text" as const, text: `result:${q}` }],
					details: {},
				};
			},
		};

		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model,
				systemPrompt: "Test",
				tools: [tool],
			},
			streamFn: async (_model, context) => {
				const stream = new MockAssistantStream();
				queueMicrotask(() => {
					const hasToolResult = context.messages.some((message) => message.role === "toolResult");

					if (hasToolResult) {
						const message: AssistantMessage = {
							role: "assistant",
							content: [{ type: "text", text: "done" }],
							api: "anthropic-messages",
							provider: "anthropic",
							model: "mock",
							usage: {
								input: 1,
								output: 1,
								cacheRead: 0,
								cacheWrite: 0,
								totalTokens: 2,
								cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
							},
							stopReason: "stop",
							timestamp: Date.now(),
						};
						stream.push({ type: "start", partial: { ...message, content: [] } });
						stream.push({ type: "done", reason: "stop", message });
						return;
					}

					const message: AssistantMessage = {
						role: "assistant",
						content: [
							{ type: "text", text: "calling tool" },
							{ type: "toolCall", id: "toolu_1", name: "dummy", arguments: { q: "x" } },
						],
						api: "anthropic-messages",
						provider: "anthropic",
						model: "mock",
						usage: {
							input: 1,
							output: 1,
							cacheRead: 0,
							cacheWrite: 0,
							totalTokens: 2,
							cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
						},
						stopReason: "toolUse",
						timestamp: Date.now(),
					};

					stream.push({ type: "start", partial: { ...message, content: [] } });
					stream.push({ type: "done", reason: "toolUse", message });
				});
				return stream;
			},
		});

		const sessionManager = SessionManager.inMemory();
		const settingsManager = SettingsManager.create(tempDir, tempDir);
		const authStorage = AuthStorage.create(join(tempDir, "auth.json"));
		const modelRegistry = ModelRegistry.create(authStorage, tempDir);
		authStorage.setRuntimeApiKey("anthropic", "test-key");

		session = new AgentSession({
			agent,
			sessionManager,
			settingsManager,
			cwd: tempDir,
			modelRegistry,
			resourceLoader: createTestResourceLoader(),
			baseToolsOverride: { dummy: tool },
		});

		const sessionWithRunner = session as unknown as {
			_extensionRunner?: {
				hasHandlers: (eventType: string) => boolean;
				emit: (event: { type: string; message?: { role?: string } }) => Promise<void>;
				emitMessageEnd: (event: { type: string; message?: { role?: string } }) => Promise<undefined>;
				emitInput: (
					text: string,
					images: unknown,
					source: "interactive" | "rpc" | "extension",
				) => Promise<{ action: "continue" }>;
				emitBeforeAgentStart: (
					prompt: string,
					images: unknown,
					systemPrompt: string,
					systemPromptOptions: BuildSystemPromptOptions,
				) => Promise<undefined>;
				invalidate: (message?: string) => void;
			};
		};
		sessionWithRunner._extensionRunner = {
			hasHandlers: () => false,
			emit: async () => {},
			emitMessageEnd: async (event) => {
				if (event.type === "message_end" && event.message?.role === "assistant") {
					await new Promise((resolve) => setTimeout(resolve, 40));
				}
				return undefined;
			},
			emitInput: async () => ({ action: "continue" }),
			emitBeforeAgentStart: async () => undefined,
			invalidate: () => {},
		};

		await session.prompt("hi");
		await session.agent.waitForIdle();
		await new Promise((resolve) => setTimeout(resolve, 100));

		const messageEntries = sessionManager.getEntries().filter((entry) => entry.type === "message");
		expect(messageEntries.map((entry) => entry.message.role)).toEqual([
			"user",
			"assistant",
			"toolResult",
			"assistant",
		]);
	});
});
