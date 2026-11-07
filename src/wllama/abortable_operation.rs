use futures::StreamExt;
use futures::channel::mpsc::{TryRecvError, UnboundedReceiver, UnboundedSender};
use futures::future::Either;
use js_sys::{Function, Promise};
use wasm_bindgen::closure::{Closure, WasmClosure};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{AbortController, AbortSignal};

use crate::Error;
use crate::wllama::wllama_bridge;

pub enum OperationEvent<Event> {
    Event(Event),
    /// The rejection is the operation's JavaScript error.
    Settled(Result<JsValue, JsValue>),
}

/// A running JavaScript operation, aborted when this is dropped, and the events its callback sends.
pub struct AbortableOperation<Event, Callback: ?Sized> {
    abort_controller: AbortController,
    settlement: JsFuture,
    event_receiver: UnboundedReceiver<Event>,
    // Freed only after the abort, since the bridge stops calling a callback once its operation is aborted.
    _event_callback: Closure<Callback>,
}

impl<Event, Callback: ?Sized + WasmClosure> AbortableOperation<Event, Callback> {
    pub fn start(
        create_event_callback: impl FnOnce(UnboundedSender<Event>) -> Closure<Callback>,
        start_operation: impl FnOnce(&AbortSignal, &Function) -> Promise,
    ) -> Result<Self, Error> {
        let abort_controller: AbortController = AbortController::new().map_err(|javascript_error: JsValue| wllama_bridge::create_error(&javascript_error))?;
        let (event_sender, event_receiver): (UnboundedSender<Event>, UnboundedReceiver<Event>) = futures::channel::mpsc::unbounded();
        let event_callback: Closure<Callback> = create_event_callback(event_sender);
        let promise: Promise = start_operation(&abort_controller.signal(), event_callback.as_ref().unchecked_ref());

        Ok(AbortableOperation {
            abort_controller,
            settlement: JsFuture::from(promise),
            event_receiver,
            _event_callback: event_callback,
        })
    }
}

impl<Event, Callback: ?Sized> AbortableOperation<Event, Callback> {
    pub fn abort_controller(&self) -> &AbortController {
        &self.abort_controller
    }

    /// Yields every event sent before the operation settled ahead of the settlement; call no more after it.
    pub async fn next(&mut self) -> OperationEvent<Event> {
        let buffered_event: Result<Event, TryRecvError> = self.event_receiver.try_recv();

        if let Ok(event) = buffered_event {
            return OperationEvent::Event(event);
        }

        let first_ready = futures::future::select(self.event_receiver.next(), &mut self.settlement).await;
        let settlement: Result<JsValue, JsValue> = match first_ready {
            Either::Left((Some(event), _)) => return OperationEvent::Event(event),
            Either::Left((None, _)) => (&mut self.settlement).await,
            Either::Right((settlement, _)) => settlement,
        };

        OperationEvent::Settled(settlement)
    }
}

impl<Event, Callback: ?Sized> Drop for AbortableOperation<Event, Callback> {
    fn drop(&mut self) {
        self.abort_controller.abort();
    }
}
