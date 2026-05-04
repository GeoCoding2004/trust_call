package com.trustcallapp

import android.Manifest
import android.content.pm.PackageManager
import android.provider.ContactsContract
import androidx.core.content.ContextCompat
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod

class TrustCallContactsModule(
    private val reactContext: ReactApplicationContext,
) : ReactContextBaseJavaModule(reactContext) {

    override fun getName(): String = "TrustCallContacts"

    @ReactMethod
    fun getContacts(promise: Promise) {
        if (
            ContextCompat.checkSelfPermission(
                reactContext,
                Manifest.permission.READ_CONTACTS,
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            promise.reject("contacts_permission_denied", "READ_CONTACTS permission is not granted.")
            return
        }

        val contacts = Arguments.createArray()
        val projection = arrayOf(
            ContactsContract.CommonDataKinds.Phone.CONTACT_ID,
            ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME,
            ContactsContract.CommonDataKinds.Phone.NUMBER,
        )
        val sortOrder = "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME} COLLATE LOCALIZED ASC"

        try {
            reactContext.contentResolver.query(
                ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
                projection,
                null,
                null,
                sortOrder,
            )?.use { cursor ->
                val idIndex = cursor.getColumnIndexOrThrow(
                    ContactsContract.CommonDataKinds.Phone.CONTACT_ID,
                )
                val nameIndex = cursor.getColumnIndexOrThrow(
                    ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME,
                )
                val numberIndex = cursor.getColumnIndexOrThrow(
                    ContactsContract.CommonDataKinds.Phone.NUMBER,
                )
                val seen = mutableSetOf<String>()

                while (cursor.moveToNext()) {
                    val id = cursor.getString(idIndex) ?: continue
                    val name = cursor.getString(nameIndex)?.trim().orEmpty()
                    val number = cursor.getString(numberIndex)?.trim().orEmpty()
                    if (name.isBlank() && number.isBlank()) continue

                    val key = "$id|$number"
                    if (!seen.add(key)) continue

                    val contact = Arguments.createMap()
                    contact.putString("id", id)
                    contact.putString("name", if (name.isBlank()) number else name)
                    contact.putString("phoneNumber", number)
                    contacts.pushMap(contact)
                }
            }

            promise.resolve(contacts)
        } catch (error: Exception) {
            promise.reject("contacts_read_failed", error.message, error)
        }
    }
}
